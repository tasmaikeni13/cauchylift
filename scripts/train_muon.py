#!/usr/bin/env python3
"""High-performance distributed pretraining for Transformer models using Muon on Google Cloud TPU v4-32.

Features:
- Configurable model scale: 125M (12 layers, 768 dim) and 350M (24 layers, 1024 dim).
- Muon optimizer with Newton-Schulz 5 orthogonalization on 2D matrices and AdamW on 1D/embedding parameters.
- Multi-core TPU data parallelism across 16 TPU v4 chips (4 hosts x 4 chips) with Torch-XLA PJRT.
- All-reduce gradient synchronization (xm.reduce_gradients) with bitwise identical rank updates.
- Scaled Dot-Product Attention in BF16 mixed precision.
- Continuous FineWeb-Edu token streaming via PackedTokenDataset.
- Cosine decay learning rate schedule with 10% warmup.
- Structured JSONL telemetry (loss, throughput, latency, MFU) and atomic checkpointing.
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

from cauchylift.baselines.muon import Muon
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


def compute_model_flops(num_params: int, seq_len: int, num_layers: int, hidden_dim: int) -> float:
    """Compute theoretical FLOPs per token forward+backward pass: 6 * N + 12 * L * H * Q."""
    # Standard approximation: 6 * N flops per token
    return 6.0 * float(num_params)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_dataset: PackedTokenDataset,
    eval_batches: int,
    dev: torch.device,
    world_size: int,
) -> tuple[float, float]:
    """Evaluate model on validation split and aggregate loss across all TPU chips."""
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

    # 1. Select Model Architecture
    model_scale = args.model_scale.lower()
    if model_scale == "350m":
        cfg = TransformerConfig(
            vocab_size=50257,
            hidden_dim=1024,
            num_layers=24,
            num_heads=16,
            intermediate_dim=2816,
            max_seq_len=args.seq_len,
            activation="swiglu",
            norm_eps=1e-5,
            tied_embeddings=True,
            attention_backend="flash",
        )
    elif model_scale == "125m":
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
    else:
        raise ValueError(f"Unknown model_scale: {model_scale}. Choose '125m' or '350m'.")

    # Deterministic model weights initialization across all ranks
    torch.manual_seed(args.seed)
    model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)
    total_params = sum(p.numel() for p in model.parameters())

    tokens_per_step = args.batch_size * args.seq_len * world_size
    total_steps = math.ceil(args.total_tokens / tokens_per_step)
    warmup_steps = int(args.warmup_fraction * total_steps)

    # 2. Setup Muon Optimizer
    optimizer = Muon(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        nesterov=args.nesterov,
        ns_steps=args.ns_steps,
        weight_decay=args.weight_decay,
        adamw_lr=args.adamw_lr,
        adamw_weight_decay=args.weight_decay,
        backend="auto",
    )
    for pg in optimizer.param_groups:
        pg.setdefault("base_lr", pg["lr"])

    # 3. Disjoint Token Streams per Rank
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
        print(f"MUON DISTRIBUTED PRETRAINING ON {world_size}x GOOGLE CLOUD TPU v4 (Seed {args.seed})")
        print(f"Model Architecture: {model_scale.upper()} ({total_params/1e6:.2f}M parameters)")
        print(f"Layers: {cfg.num_layers} | Hidden: {cfg.hidden_dim} | Heads: {cfg.num_heads} | Vocab: {cfg.vocab_size}")
        print(f"Total Budget: {args.total_tokens:,} tokens | Steps: {total_steps:,} | Warmup: {warmup_steps:,}")
        print(f"Effective Batch: {tokens_per_step:,} tokens/step ({args.batch_size} micro-batch x {world_size} chips)")
        print(f"Muon LR: {args.lr:.2e} | Momentum: {args.momentum} | Weight Decay: {args.weight_decay}")
        print(f"AdamW Auxiliary LR: {args.adamw_lr:.2e} | NS Steps: {args.ns_steps} | Nesterov: {args.nesterov}")
        print("=" * 80)
        sys.stdout.flush()

    start_step = 1
    tokens_seen = 0
    best_val_loss = float("inf")

    # Resumption support
    best_ckpt = ckpt_dir / "best.pt"
    if best_ckpt.exists():
        try:
            b_ckpt = torch.load(str(best_ckpt), map_location="cpu")
            best_val_loss = b_ckpt.get("val_loss", float("inf"))
        except Exception:
            pass

    latest_ckpt = ckpt_dir / "latest.pt"
    if latest_ckpt.exists():
        try:
            checkpoint = torch.load(str(latest_ckpt), map_location="cpu")
            model.load_state_dict({k: v.to(device=dev, dtype=torch.bfloat16) for k, v in checkpoint["model_state_dict"].items()})
            start_step = checkpoint["step"] + 1
            tokens_seen = checkpoint["tokens_seen"]
            best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
            if rank == 0:
                print(f">>> [RESUMED] Resumed training from checkpoint at step {checkpoint['step']} ({tokens_seen:,} tokens seen, best_val_loss={best_val_loss:.4f})")
                sys.stdout.flush()
        except Exception as e:
            if rank == 0:
                print(f"Warning: Failed to load checkpoint {latest_ckpt}: {e}. Starting from scratch.")
                sys.stdout.flush()

    model.train()
    t_global_start = time.perf_counter()
    recent_step_times = []
    flops_per_token = compute_model_flops(total_params, args.seq_len, cfg.num_layers, cfg.hidden_dim)

    for step in range(start_step, total_steps + 1):
        step_t0 = time.perf_counter()

        # Update per-group learning rate with cosine decay
        for pg in optimizer.param_groups:
            base_lr = pg["base_lr"]
            min_lr = base_lr * 0.1
            pg["lr"] = get_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr)

        optimizer.zero_grad()

        x_cpu, y_cpu, _ = train_dataset.next_batch()
        x = x_cpu.to(device=dev)
        y = y_cpu.to(device=dev)

        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)

        loss.backward()

        # All-reduce gradients across all TPU ranks
        xm.reduce_gradients(optimizer)

        # Muon step on TPU
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
            # TPU v4 peak FLOPs per chip: 275 TFLOPs
            achieved_tflops = (tok_per_sec * flops_per_token) / 1e12
            peak_cluster_tflops = 275.0 * world_size
            mfu = (achieved_tflops / peak_cluster_tflops) * 100.0
            elapsed_min = (time.perf_counter() - t_global_start) / 60.0

            muon_lr_now = optimizer.param_groups[0]["lr"]
            adamw_lr_now = optimizer.param_groups[1]["lr"] if len(optimizer.param_groups) > 1 else muon_lr_now

            print(
                f"[Step {step:5d}/{total_steps}] "
                f"Loss: {loss_val:.4f} | "
                f"Tokens: {tokens_seen:,} ({tokens_seen/args.total_tokens*100:.1f}%) | "
                f"MuonLR: {muon_lr_now:.2e} | "
                f"Throughput: {tok_per_sec:,.0f} tok/s | "
                f"Latency: {avg_duration*1000:.1f}ms | "
                f"MFU: {mfu:.1f}% | "
                f"Elapsed: {elapsed_min:.1f}m"
            )
            sys.stdout.flush()

            with open(log_file, "a") as f:
                f.write(json.dumps({
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "train_loss": loss_val,
                    "muon_lr": muon_lr_now,
                    "adamw_lr": adamw_lr_now,
                    "step_duration_s": avg_duration,
                    "tokens_per_sec": tok_per_sec,
                    "achieved_tflops": achieved_tflops,
                    "mfu_percent": mfu,
                    "elapsed_min": elapsed_min,
                    "timestamp": time.time(),
                }) + "\n")

        # Periodic validation
        if step % args.eval_interval == 0 or step == total_steps:
            val_loss, ppl = evaluate(model, val_dataset, args.eval_batches, dev, world_size)
            if rank == 0:
                print("-" * 80)
                print(f">>> [EVAL step {step:5d}] Validation Loss: {val_loss:.4f} | Perplexity: {ppl:.2f}")
                print("-" * 80)
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = ckpt_dir / "best.pt"
                best_ckpt_data = {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "val_loss": val_loss,
                    "perplexity": ppl,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg.__dict__,
                    "args": vars(args),
                }
                xm.save(best_ckpt_data, str(best_path), master_only=True, global_master=True)
                xm.rendezvous(f"best_ckpt_{step}")
                if rank == 0:
                    print(f">>> [CHECKPOINT] Saved new best model (val_loss: {val_loss:.4f}) to {best_path}")
                    sys.stdout.flush()

        # Periodic checkpointing
        if step % args.checkpoint_interval == 0 or step == total_steps:
            latest_path = ckpt_dir / "latest.pt"
            ckpt_data = {
                "step": step,
                "tokens_seen": tokens_seen,
                "best_val_loss": best_val_loss,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg.__dict__,
                "args": vars(args),
            }
            xm.save(ckpt_data, str(latest_path), master_only=True, global_master=True)
            xm.rendezvous(f"latest_ckpt_{step}")
            if rank == 0:
                print(f">>> [CHECKPOINT] Saved latest checkpoint at step {step} to {latest_path}")
                sys.stdout.flush()

    xm.rendezvous("pretraining_complete")
    if rank == 0:
        total_time_min = (time.perf_counter() - t_global_start) / 60.0
        print("=" * 80)
        print("PRETRAINING COMPLETE FOR MUON!")
        print(f"Model Scale: {model_scale.upper()} ({total_params/1e6:.2f}M params)")
        print(f"Total Tokens: {tokens_seen:,} | Total Time: {total_time_min:.2f} minutes | Best Val Loss: {best_val_loss:.4f}")
        print("=" * 80)
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Distributed Pretraining for Transformer using Muon on Google Cloud TPU v4-32")
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m"])
    parser.add_argument("--total_tokens", type=int, default=3_000_000_000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.02, help="Muon learning rate for 2D internal matrices")
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--nesterov", action="store_true", default=True)
    parser.add_argument("--ns_steps", type=int, default=5, help="Newton-Schulz quintic iteration steps")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adamw_lr", type=float, default=6e-4, help="AdamW learning rate for 1D and embedding weights")
    parser.add_argument("--warmup_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="runs/muon_125m_seed42")
    args = parser.parse_args()

    xmp.spawn(_train_rank, args=(args,))


if __name__ == "__main__":
    main()
