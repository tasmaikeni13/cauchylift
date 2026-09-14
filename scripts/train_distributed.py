#!/usr/bin/env python3
"""High-performance distributed pretraining for 125M Transformer on Google Cloud TPU v4-32.

Orchestrates multi-core data parallelism across Google Cloud TPU chips
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


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    momentum: float,
    weight_decay: float,
    adamw_lr: float = 6e-4,
    ns_steps: int = 5,
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
            adamw_lr=adamw_lr,
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
    elif opt_lower == "muon":
        return Muon(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_weight_decay=weight_decay,
            ns_steps=ns_steps,
            backend="auto",
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

    model_scale = getattr(args, "model_scale", "125m").lower()
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
    else:
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

    optimizer = build_optimizer(
        model,
        args.optimizer,
        args.lr,
        args.momentum,
        args.weight_decay,
        adamw_lr=getattr(args, "adamw_lr", 6e-4),
        ns_steps=getattr(args, "ns_steps", 5),
    )
    for pg in optimizer.param_groups:
        pg.setdefault("base_lr", pg["lr"])

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
        tpu_type = "TPU v4"
        print("=" * 80)
        print(f"Distributed Pretraining on {world_size}x {tpu_type}: {args.optimizer.upper()} (Seed {args.seed})")
        print(f"Model: {model_scale.upper()} ({total_params/1e6:.1f}M params) | Vocab: {cfg.vocab_size} | SeqLen: {args.seq_len}")
        print(f"Total Budget: {args.total_tokens:,} tokens | Steps: {total_steps:,} | Warmup: {warmup_steps:,}")
        print(f"Effective Batch: {tokens_per_step:,} tokens/step ({args.batch_size} micro-batch x {world_size} chips)")
        print(f"Base LR: {args.lr:.2e} | Momentum: {args.momentum} | Weight Decay: {args.weight_decay}")
        if args.optimizer.lower() == "muon":
            print(f"Muon AdamW LR: {getattr(args, 'adamw_lr', 6e-4):.2e} | NS Steps: {getattr(args, 'ns_steps', 5)}")
    start_step = 1
    tokens_seen = 0
    best_val_loss = float("inf")
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

    flops_per_token = 6.0 * float(total_params)

    for step in range(start_step, total_steps + 1):
        step_t0 = time.perf_counter()
        for pg in optimizer.param_groups:
            base_lr = pg["base_lr"]
            min_lr = base_lr * 0.1
            pg["lr"] = get_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr)
        current_lr = optimizer.param_groups[0]["lr"]

        optimizer.zero_grad()

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
        if rank == 0 and (step % args.log_interval == 0 or step == total_steps):
            loss_val = float(loss.item())
            avg_duration = sum(recent_step_times) / len(recent_step_times)
            tok_per_sec = tokens_per_step / max(avg_duration, 1e-6)
            achieved_tflops = (tok_per_sec * flops_per_token) / 1e12
            mfu = (achieved_tflops / (275.0 * world_size)) * 100.0
            elapsed_s = time.perf_counter() - t_global_start
            remaining_steps = total_steps - step
            eta_s = remaining_steps * avg_duration
            eta_min = eta_s / 60.0

            print(
                f"[Step {step:5d}/{total_steps:5d}] "
                f"Loss: {loss_val:.4f} | "
                f"LR: {current_lr:.2e} | "
                f"Speed: {tok_per_sec:,.0f} tok/s ({avg_duration*1000:.1f} ms) | "
                f"MFU: {mfu:4.1f}% | "
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
                    "mfu": mfu,
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

        # Periodic checkpointing (only saves latest.pt to conserve storage)
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
        avg_step_ms = sum(recent_step_times) / max(1, len(recent_step_times)) * 1000.0
        steady_tok_s = tokens_per_step / max(1e-6, avg_step_ms / 1000.0)
        mean_mfu = ((steady_tok_s * flops_per_token) / 1e12) / (275.0 * world_size) * 100.0

        run_summary = {
            "optimizer": args.optimizer,
            "seed": args.seed,
            "base_lr": args.lr,
            "total_tokens": tokens_seen,
            "total_steps": total_steps,
            "best_val_loss": best_val_loss,
            "final_train_loss": loss_val,
            "tokens_per_sec": steady_tok_s,
            "mfu": mean_mfu,
            "elapsed_minutes": total_time_min,
        }
        with open(out_dir / "run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)

        print("=" * 80)
        print(f"Pretraining Complete for {args.optimizer.upper()}!")
        print(f"Total Tokens: {tokens_seen:,} | Total Time: {total_time_min:.2f} minutes")
        print(f"Best Val Loss: {best_val_loss:.4f} | Speed: {steady_tok_s:,.0f} tok/s | MFU: {mean_mfu:4.1f}%")
        print(f"Saved run summary to {out_dir / 'run_summary.json'}")
        print("=" * 80)
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Distributed Multi-Chip Pretraining on Google Cloud TPU v4-32")
    parser.add_argument("--optimizer", type=str, default="cauchylift", choices=["cauchylift", "adamw", "muon"])
    parser.add_argument("--model_scale", type=str, default="125m", choices=["125m", "350m"])
    parser.add_argument("--total_tokens", type=int, default=2_500_000_000)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adamw_lr", type=float, default=6e-4)
    parser.add_argument("--ns_steps", type=int, default=5)
    parser.add_argument("--warmup_fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="runs/cauchylift_125m_seed42")
    args = parser.parse_args()


    # Launch across available TPU cores
    xmp.spawn(_train_rank, args=(args,))
    os._exit(0)


if __name__ == "__main__":
    main()
