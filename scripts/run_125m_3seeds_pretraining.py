#!/usr/bin/env python3
"""Run 3 Seeds per Optimizer (CauchyLift, AdamW) for 125M on 2.5B FineWeb-Edu Tokens.

Orchestrates 6 complete pretraining runs across all 16 Google Cloud TPU v4 chips:
- CauchyLift (Seeds 42, 43, 44) | LR: 0.0010, Momentum: 0.95, WD: 0.01, AdamW LR: 6e-4
- AdamW (Seeds 42, 43, 44)      | LR: 0.0020, Betas: (0.9, 0.95), WD: 0.01

Pretraining Protocol (experiments/protocols/protocol_125m_fineweb.json):
- Model: 125M Decoder Transformer (768 dim, 12 layers, 12 heads, SwiGLU)
- Total Tokens: 2,500,000,000 tokens
- Effective Batch Tokens: 262,144 tokens/step (batch_size=8, 16 chips, seq_len=2048)
- Optimization Steps: 9,537 steps
- Warmup Steps: 954 steps (10%)
- Schedule: Cosine decay to 0.1x LR
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pathlib
import sys
import time
import traceback
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
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    if step > total_steps:
        return min_lr
    decay_ratio = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (base_lr - min_lr)


def _build_optimizer(
    model: nn.Module,
    opt_name: str,
    base_lr: float,
    wd: float = 0.01,
    adamw_lr: float = 0.0006,
) -> torch.optim.Optimizer:
    if opt_name == "cauchylift":
        decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        opt = CauchyLift(param_groups, lr=base_lr, momentum=0.95, weight_decay=wd, backend="auto")
    elif opt_name == "adamw":
        decay = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
        param_groups = [
            {"params": decay, "weight_decay": wd},
            {"params": nodecay, "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
    elif opt_name == "muon":
        opt = Muon(
            model.parameters(),
            lr=base_lr,
            momentum=0.95,
            nesterov=True,
            ns_steps=5,
            weight_decay=wd,
            adamw_lr=adamw_lr,
            adamw_weight_decay=wd,
            backend="auto",
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")

    for pg in opt.param_groups:
        pg.setdefault("base_lr", pg["lr"])
    return opt


def _execute_single_run(
    dev: torch.device,
    rank: int,
    world_size: int,
    opt_name: str,
    seed: int,
    base_lr: float,
    total_tokens: int,
    seq_len: int,
    batch_size: int,
    output_dir: pathlib.Path,
    adamw_lr: float = 0.0006,
):
    tokens_per_opt_step = batch_size * seq_len * world_size
    total_opt_steps = math.ceil(total_tokens / tokens_per_opt_step)
    warmup_steps = max(1, int(0.10 * total_opt_steps))

    cfg = TransformerConfig(
        vocab_size=50257,
        hidden_dim=768,
        num_layers=12,
        num_heads=12,
        intermediate_dim=2048,
        max_seq_len=seq_len,
        activation="swiglu",
        norm_eps=1e-5,
        tied_embeddings=True,
        attention_backend="flash",
    )

    torch.manual_seed(seed)
    model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)
    opt = _build_optimizer(model, opt_name, base_lr, adamw_lr=adamw_lr)

    dataset = PackedTokenDataset(
        split="train",
        max_seq_len=seq_len,
        batch_size=batch_size,
        seed=seed + rank * 100000,
    )

    val_dataset = PackedTokenDataset(
        split="validation",
        max_seq_len=seq_len,
        batch_size=batch_size,
        seed=12345 + rank * 100000,
    )

    total_params = sum(p.numel() for p in model.parameters())
    flops_per_token = 6.0 * float(total_params)

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 80)
        print(f"STARTING RUN: {opt_name.upper()} | SEED {seed}")
        print(f"Target Budget: {total_tokens:,} tokens | Steps: {total_opt_steps:,} | Warmup: {warmup_steps:,}")
        print(f"Base LR: {base_lr} | Batch/chip: {batch_size} | Effective Batch Tokens: {tokens_per_opt_step:,}")
        print(f"Output Directory: {output_dir.resolve()}")
        print("=" * 80)
        sys.stdout.flush()

    tokens_seen = 0
    t_start = time.perf_counter()
    step_times = []
    best_val_loss = float("inf")

    model.train()

    for opt_step in range(1, total_opt_steps + 1):
        step_t0 = time.perf_counter()

        # Cosine LR decay
        for pg in opt.param_groups:
            b_lr = pg["base_lr"]
            m_lr = b_lr * 0.1
            pg["lr"] = get_cosine_lr(opt_step, warmup_steps, total_opt_steps, b_lr, m_lr)

        opt.zero_grad()

        x_cpu, y_cpu, _ = dataset.next_batch()
        x = x_cpu.to(device=dev)
        y = y_cpu.to(device=dev)

        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)

        loss.backward()
        xm.reduce_gradients(opt)
        opt.step()
        torch_xla.sync()

        step_dur = time.perf_counter() - step_t0
        step_times.append(step_dur)
        tokens_seen += tokens_per_opt_step

        # Uniform step-wise collective all-reduce maintains consistent XLA graph
        global_loss = xm.all_reduce("sum", loss.to(torch.float32)) / float(world_size)
        global_loss_val = float(global_loss.item())

        # Periodic logging on rank 0
        if opt_step == 1 or opt_step % 20 == 0 or opt_step == total_opt_steps:
            steady_times = step_times[-20:]
            avg_step_ms = (sum(steady_times) / max(1, len(steady_times))) * 1000.0
            tok_per_sec = tokens_per_opt_step / max(1e-6, avg_step_ms / 1000.0)
            achieved_tflops = (tok_per_sec * flops_per_token) / 1e12
            mfu = (achieved_tflops / (275.0 * world_size)) * 100.0
            elapsed_m = (time.perf_counter() - t_start) / 60.0
            remaining_steps = total_opt_steps - opt_step
            eta_m = (remaining_steps * (avg_step_ms / 1000.0)) / 60.0

            if rank == 0:
                print(
                    f"[{opt_name.upper()} s{seed}] Step {opt_step:5d}/{total_opt_steps:5d} | "
                    f"Loss: {global_loss_val:.4f} | "
                    f"LR: {opt.param_groups[0]['lr']:.2e} | "
                    f"Tok/s: {tok_per_sec:,.0f} | "
                    f"MFU: {mfu:4.1f}% | "
                    f"Seen: {tokens_seen/1e9:.3f}B ({tokens_seen/total_tokens*100:4.1f}%) | "
                    f"Elapsed: {elapsed_m:5.1f}m | ETA: {eta_m:5.1f}m",
                    flush=True,
                )

        # Periodic validation
        if opt_step % 500 == 0 or opt_step == total_opt_steps:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for _ in range(5):
                    vx, vy, _ = val_dataset.next_batch()
                    with torch.autocast(device_type="xla", dtype=torch.bfloat16):
                        _, vloss = model(vx.to(dev), vy.to(dev))
                    val_losses.append(vloss.to(torch.float32))
            avg_val_t = sum(val_losses) / float(len(val_losses))
            global_val = float((xm.all_reduce("sum", avg_val_t) / float(world_size)).item())
            model.train()

            if rank == 0:
                ppl = math.exp(min(20.0, global_val))
                print(f">>> [VAL @ Step {opt_step}] Loss: {global_val:.4f} | PPL: {ppl:.2f}", flush=True)

                if global_val < best_val_loss:
                    best_val_loss = global_val
                    xm.save(
                        {"step": opt_step, "val_loss": global_val, "state_dict": model.state_dict()},
                        str(output_dir / "best.pt"),
                        master_only=True,
                        global_master=True,
                    )

        # Periodic checkpointing
        if opt_step % 2000 == 0 or opt_step == total_opt_steps:
            if rank == 0:
                xm.save(
                    {"step": opt_step, "tokens_seen": tokens_seen, "loss": global_loss_val, "state_dict": model.state_dict()},
                    str(output_dir / "latest.pt"),
                    master_only=True,
                    global_master=True,
                )

    total_m = (time.perf_counter() - t_start) / 60.0
    if rank == 0:
        print("\n" + "=" * 80)
        print(f"RUN COMPLETED: {opt_name.upper()} | SEED {seed}")
        print(f"Total Tokens Evaluated: {tokens_seen:,} | Total Time: {total_m:.1f} minutes")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print("=" * 80)
        sys.stdout.flush()

        run_summary = {
            "optimizer": opt_name,
            "seed": seed,
            "base_lr": base_lr,
            "total_tokens": tokens_seen,
            "total_steps": total_opt_steps,
            "best_val_loss": best_val_loss,
            "elapsed_minutes": total_m,
        }
        with open(output_dir / "run_summary.json", "w") as f:
            json.dump(run_summary, f, indent=2)

    del model, opt, dataset, val_dataset
    gc.collect()


def _pretrain_all_main(index: int, args: argparse.Namespace):
    try:
        dev = xm.xla_device()
        rank = xr.global_ordinal()
        world_size = xr.world_size()

        protocol_path = pathlib.Path("experiments/protocols/protocol_125m_fineweb.json")
        with open(protocol_path) as f:
            protocol = json.load(f)

        opt_configs = protocol["optimizers"]
        seeds = args.seeds if args.seeds else protocol.get("seeds", [42, 43, 44])
        selected_opts = [args.optimizer] if args.optimizer != "all" else ["cauchylift", "adamw"]

        runs = []
        for opt_n in selected_opts:
            for s in seeds:
                runs.append({
                    "optimizer": opt_n,
                    "seed": s,
                    "lr": opt_configs[opt_n]["lr"],
                    "adamw_lr": opt_configs[opt_n].get("adamw_lr", 0.0006),
                })

        if rank == 0:
            print("=" * 80)
            print("DISTRIBUTED PRETRAINING: 3 SEEDS x 2 OPTIMIZERS (125M on 2.5B FineWeb-Edu TOKENS)")
            print(f"Hardware: {world_size}x Google Cloud TPU v4 chips (TPU v4-32 slice, 4 nodes)")
            print(f"Protocol: {protocol_path}")
            print(f"Total Scheduled Runs: {len(runs)}")
            for r in runs:
                print(f"  - {r['optimizer'].upper()} | Seed {r['seed']} | LR: {r['lr']}")
            print("=" * 80)
            sys.stdout.flush()

        for idx, run_cfg in enumerate(runs):
            opt_name = run_cfg["optimizer"]
            seed = run_cfg["seed"]
            lr = run_cfg["lr"]
            adamw_lr = run_cfg.get("adamw_lr", 0.0006)
            output_dir = pathlib.Path(f"runs/125m_{opt_name}_seed{seed}")

            if rank == 0:
                print(f"\n>>> [QUEUE {idx+1}/{len(runs)}] Launching {opt_name.upper()} (Seed {seed}, LR {lr}) on all {world_size} TPU chips...")
                sys.stdout.flush()

            _execute_single_run(
                dev=dev,
                rank=rank,
                world_size=world_size,
                opt_name=opt_name,
                seed=seed,
                base_lr=lr,
                total_tokens=args.total_tokens,
                seq_len=2048,
                batch_size=8,
                output_dir=output_dir,
                adamw_lr=adamw_lr,
            )

            torch_xla.sync()

        if rank == 0:
            print("\n" + "=" * 80)
            print("ALL PRETRAINING RUNS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            sys.stdout.flush()
    except Exception as e:
        rank = xr.global_ordinal() if xr is not None else index
        err_msg = f"\n[CRITICAL ERROR ON RANK {rank}]: {e}\n{traceback.format_exc()}\n"
        print(err_msg, file=sys.stderr, flush=True)
        print(err_msg, file=sys.stdout, flush=True)
        with open(f"/tmp/tpu_rank_{rank}_error.log", "w") as f:
            f.write(err_msg)
        os._exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run 3 seeds x 2 optimizers for 125M on 2.5B tokens")
    parser.add_argument("--total_tokens", type=int, default=2_500_000_000, help="Total token budget per run (default: 2.5B)")
    parser.add_argument("--optimizer", type=str, default="all", choices=["cauchylift", "adamw", "all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44], help="Random seeds to evaluate")
    args = parser.parse_args()

    xmp.spawn(_pretrain_all_main, args=(args,))
    os._exit(0)


if __name__ == "__main__":
    main()
