#!/usr/bin/env python3
"""High-performance distributed 8-chip pretraining for 125M Transformer on 3B tokens.

Orchestrates multi-core data parallelism across 8x Google Cloud TPU v6e (Trillium)
using Torch-XLA PJRT distributed runtime with all-reduce gradient synchronization,
mixed precision BF16, and periodic validation and checkpointing.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from typing import Any

import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from cauchylift import CauchyLift
from cauchylift.data import PackedTokenDataset
from cauchylift.models.transformer import Transformer, TransformerConfig


def get_cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float, min_lr: float) -> float:
    """Compute learning rate with linear warmup and cosine decay."""
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    if step > total_steps:
        return min_lr
    decay_ratio = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    decay_params = []
    nodecay_params = []
    for p in model.parameters():
        if p.requires_grad:
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    opt_lower = optimizer_name.lower()
    if opt_lower == "cauchylift":
        return CauchyLift(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            backend="auto",
        )
    elif opt_lower == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=lr,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=weight_decay,
            fused=False,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_dataset: PackedTokenDataset,
    eval_batches: int,
    dev: torch.device,
    world_size: int,
) -> tuple[float, float]:
    model.eval()
    total_local = torch.tensor(0.0, device=dev, dtype=torch.float32)
    for _ in range(eval_batches):
        x_cpu, y_cpu, _ = val_dataset.next_batch()
        x = x_cpu.to(device=dev)
        y = y_cpu.to(device=dev)
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)
        total_local += loss.to(torch.float32)

    torch_xla.sync()
    avg_local = total_local / float(eval_batches)
    global_loss_t = xm.all_reduce("sum", avg_local) / float(world_size)
    global_loss = float(global_loss_t.item())
    ppl = math.exp(min(global_loss, 20.0))
    model.train()
    return global_loss, ppl


def _train_rank(index: int, args: argparse.Namespace):
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    world_size = xr.world_size()

    # Model architecture (125M decoder-only Transformer)
    cfg = TransformerConfig(
        vocab_size=50257,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=2048,
        max_seq_len=args.seq_len,
        activation="swiglu",
        norm_eps=1e-5,
        tied_embeddings=True,
        attention_backend="flash",
    )

    # Deterministic model weights initialization across all ranks
    torch.manual_seed(args.seed)
    model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())

    tokens_per_step = args.batch_size * args.seq_len * world_size
    total_steps = math.ceil(args.total_tokens / tokens_per_step)
    warmup_steps = int(args.warmup_fraction * total_steps)
    min_lr = args.lr * 0.1

    optimizer = build_optimizer(model, args.optimizer, args.lr, args.momentum, args.weight_decay)

    # Disjoint token streams per rank
    train_dataset = PackedTokenDataset(
        split="train",
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed + rank * 100000,
    )
    val_dataset = PackedTokenDataset(
        split="validation",
        max_seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=9999 + rank * 1000,
    )

    out_dir = pathlib.Path(args.output_dir)
    log_file = out_dir / "metrics.jsonl"
    ckpt_dir = out_dir / "checkpoints"

    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        print("=" * 80)
        print(f"Distributed Pretraining on 8x TPU v6e: {args.optimizer.upper()} (Seed {args.seed})")
        print(f"Model: 125M ({total_params/1e6:.1f}M params) | Vocab: {cfg.vocab_size} | SeqLen: {args.seq_len}")
        print(f"Total Budget: {args.total_tokens:,} tokens | Steps: {total_steps:,} | Warmup: {warmup_steps:,}")
        print(f"Effective Batch: {tokens_per_step:,} tokens/step ({args.batch_size} micro-batch x {world_size} chips)")
        print(f"Base LR: {args.lr:.2e} | Min LR: {min_lr:.2e} | Momentum: {args.momentum} | Weight Decay: {args.weight_decay}")
    start_step = 1
    tokens_seen = 0
    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        try:
            checkpoint = torch.load(str(latest_ckpt), map_location="cpu")
            model.load_state_dict({k: v.to(device=dev, dtype=torch.bfloat16) for k, v in checkpoint["model_state_dict"].items()})
            start_step = checkpoint["step"] + 1
            tokens_seen = checkpoint["tokens_seen"]
            if rank == 0:
                print(f">>> [RESUMED] Resumed training from checkpoint at step {checkpoint['step']} ({tokens_seen:,} tokens seen)")
                sys.stdout.flush()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to load checkpoint {latest_ckpt}: {e}. Starting from scratch.")
                sys.stdout.flush()

    model.train()
    t_global_start = time.perf_counter()
    recent_step_times = []

    for step in range(start_step, total_steps + 1):
        step_t0 = time.perf_counter()
        current_lr = get_cosine_lr(step, warmup_steps, total_steps, args.lr, min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.zero_grad(set_to_none=True)

        x_cpu, y_cpu, _ = train_dataset.next_batch()
        x = x_cpu.to(device=dev)
        y = y_cpu.to(device=dev)

        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)

        loss.backward()
        xm.reduce_gradients(optimizer)
        optimizer.step()
        torch_xla.sync()

        tokens_seen += tokens_per_step
        step_duration = time.perf_counter() - step_t0
        recent_step_times.append(step_duration)
        if len(recent_step_times) > 50:
            recent_step_times.pop(0)

        # Periodic logging on rank 0
        if rank == 0 and (step % args.log_interval == 0 or step == total_steps or step <= 5):
            loss_val = float(loss.item())
            avg_duration = sum(recent_step_times) / len(recent_step_times)
            tok_per_sec = tokens_per_step / max(avg_duration, 1e-6)
            elapsed_s = time.perf_counter() - t_global_start
            remaining_steps = total_steps - step
            eta_s = remaining_steps * avg_duration
            eta_min = eta_s / 60.0

            print(
                f"[Step {step:5d}/{total_steps:5d}] "
                f"Loss: {loss_val:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {tok_per_sec:,.0f} tok/s ({avg_duration*1000:.1f} ms) | "
                f"Progress: {100.0 * tokens_seen / args.total_tokens:5.1f}% | "
                f"ETA: {eta_min:.1f}m"
            )
            sys.stdout.flush()

            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": loss_val,
                    "lr": current_lr,
                    "step_time_ms": avg_duration * 1000.0,
                    "tokens_per_sec": tok_per_sec,
                    "elapsed_s": elapsed_s,
                    "eta_minutes": eta_min,
                }) + "\n")

        # Periodic evaluation
        if step % args.eval_interval == 0 or step == total_steps:
            val_loss, ppl = evaluate(model, val_dataset, args.eval_batches, dev, world_size)
            if rank == 0:
                print(f">>> [EVAL @ Step {step:5d}] Val Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
                sys.stdout.flush()
                with open(log_file, "a") as f:
                    f.write(json.dumps({
                        "event": "eval",
                        "step": step,
                        "tokens_seen": tokens_seen,
                        "val_loss": val_loss,
                        "perplexity": ppl,
                        "timestamp": time.time(),
                    }) + "\n")

        # Periodic checkpointing
        if rank == 0 and (step % args.checkpoint_interval == 0 or step == total_steps):
            ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
            latest_path = ckpt_dir / "latest.pt"
            cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
            cpu_opt_state = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in optimizer.state_dict().items()}
            ckpt_data = {
                "step": step,
                "tokens_seen": tokens_seen,
                "model_state_dict": cpu_model_state,
                "optimizer_state_dict": cpu_opt_state,
                "config": cfg.__dict__,
                "args": vars(args),
            }
            torch.save(ckpt_data, str(ckpt_path))
            torch.save(ckpt_data, str(latest_path))
            print(f">>> [CHECKPOINT] Saved checkpoint at step {step} to {ckpt_path}")
            sys.stdout.flush()

    if rank == 0:
        total_time_min = (time.perf_counter() - t_global_start) / 60.0
        print("=" * 80)
        print(f"Pretraining Complete for {args.optimizer.upper()}!")
        print(f"Total Tokens: {tokens_seen:,} | Total Time: {total_time_min:.2f} minutes")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Distributed 8-Chip Pretraining on TPU v6e")
    parser.add_argument("--optimizer", type=str, default="cauchylift", choices=["cauchylift", "adamw"])
    parser.add_argument("--total_tokens", type=int, default=3_000_000_000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="runs/cauchylift_125m_seed42")
    args = parser.parse_args()

    # Launch across 8 TPU cores
    xmp.spawn(_train_rank, args=(args,))


if __name__ == "__main__":
    main()
