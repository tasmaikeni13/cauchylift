#!/usr/bin/env python3
"""Rigorous Hyperparameter Sweep Runner on Google Cloud TPU v4-32.

Tuning CauchyLift and AdamW on 125M Transformer (FineWeb-Edu) with 200M tokens per arm
(~763 steps @ 262,144 tok/step, ~3.2 minutes per arm across 16 TPU v4 chips).

Fair Search Grid:
- CauchyLift (2D hidden matrix base LR; adamw_lr=0.0006 for 1D/lookup):
  LRs: [0.0010, 0.0025, 0.0050, 0.0100] across seeds [42, 43, 44]
- AdamW Baseline:
  LRs: [0.0003, 0.0006, 0.0012] across seeds [42, 43, 44]

Outputs:
- runs/sweeps/{optimizer}_lr{lr:.4f}_seed{seed}/
- artifacts/sweeps/hp_sweep_results.json
- artifacts/sweeps/hp_sweep_report.md
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from launch_v4_32_distributed import (
    cleanup_dangling_tpu_processes,
    launch_distributed_command,
    sync_codebase_to_workers,
    sync_runs_back_from_workers,
)

HP_SWEEP_CONFIGS = [
    # CauchyLift Hyperparameter Grid
    {"optimizer": "cauchylift", "lr": 0.0010, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "cauchylift", "lr": 0.0010, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "cauchylift", "lr": 0.0010, "adamw_lr": 0.0006, "seed": 44},

    {"optimizer": "cauchylift", "lr": 0.0025, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "cauchylift", "lr": 0.0025, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "cauchylift", "lr": 0.0025, "adamw_lr": 0.0006, "seed": 44},

    {"optimizer": "cauchylift", "lr": 0.0050, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "cauchylift", "lr": 0.0050, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "cauchylift", "lr": 0.0050, "adamw_lr": 0.0006, "seed": 44},

    {"optimizer": "cauchylift", "lr": 0.0100, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "cauchylift", "lr": 0.0100, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "cauchylift", "lr": 0.0100, "adamw_lr": 0.0006, "seed": 44},

    # AdamW Baseline Grid
    {"optimizer": "adamw", "lr": 0.0003, "adamw_lr": 0.0003, "seed": 42},
    {"optimizer": "adamw", "lr": 0.0003, "adamw_lr": 0.0003, "seed": 43},
    {"optimizer": "adamw", "lr": 0.0003, "adamw_lr": 0.0003, "seed": 44},

    {"optimizer": "adamw", "lr": 0.0006, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "adamw", "lr": 0.0006, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "adamw", "lr": 0.0006, "adamw_lr": 0.0006, "seed": 44},

    {"optimizer": "adamw", "lr": 0.0010, "adamw_lr": 0.0010, "seed": 42},
    {"optimizer": "adamw", "lr": 0.0010, "adamw_lr": 0.0010, "seed": 43},
    {"optimizer": "adamw", "lr": 0.0010, "adamw_lr": 0.0010, "seed": 44},

    {"optimizer": "adamw", "lr": 0.0020, "adamw_lr": 0.0020, "seed": 42},
    {"optimizer": "adamw", "lr": 0.0020, "adamw_lr": 0.0020, "seed": 43},
    {"optimizer": "adamw", "lr": 0.0020, "adamw_lr": 0.0020, "seed": 44},
]


def is_arm_completed(out_dir: pathlib.Path, target_tokens: int = 200_000_000) -> bool:
    summary_file = out_dir / "run_summary.json"
    if not summary_file.exists():
        sync_runs_back_from_workers()
    if not summary_file.exists():
        return False
    try:
        with open(summary_file) as f:
            data = json.load(f)
        return int(data.get("total_tokens", 0)) >= target_tokens
    except Exception:
        return False


def run_hp_sweep(
    total_tokens: int = 200_000_000,
    batch_size: int = 8,
    seq_len: int = 2048,
    eval_interval: int = 250,
    filter_opt: str | None = None,
    filter_lr: float | None = None,
    filter_seed: int | None = None,
):
    print("\n" + "=" * 80)
    print(f"STARTING FAIR HYPERPARAMETER TUNING SWEEP ({total_tokens:,} TOKENS / ARM)")
    print(f"TPU Cluster: 16 chips (4 hosts) | Global batch size: {batch_size * 16 * seq_len:,} tok/step")
    print("=" * 80 + "\n")

    sync_codebase_to_workers()

    selected_runs = []
    for cfg in HP_SWEEP_CONFIGS:
        if filter_opt and cfg["optimizer"] != filter_opt:
            continue
        if filter_lr and abs(cfg["lr"] - filter_lr) > 1e-6:
            continue
        if filter_seed and cfg["seed"] != filter_seed:
            continue
        selected_runs.append(cfg)

    print(f"Total Arms in Queue: {len(selected_runs)}")
    for idx, r in enumerate(selected_runs, 1):
        print(f"  [{idx:2d}/{len(selected_runs)}] {r['optimizer'].upper():10s} | LR: {r['lr']:.4f} | Seed: {r['seed']}")
    print("-" * 80 + "\n")

    t_sweep_start = time.perf_counter()

    for idx, run_cfg in enumerate(selected_runs, 1):
        opt = run_cfg["optimizer"]
        lr = run_cfg["lr"]
        adamw_lr = run_cfg["adamw_lr"]
        seed = run_cfg["seed"]

        run_name = f"{opt}_lr{lr:.4f}_seed{seed}"
        out_dir = pathlib.Path(f"runs/sweeps/{run_name}")

        print("\n" + "#" * 80)
        print(f"SWEEP ARM [{idx}/{len(selected_runs)}]: Launching {run_name} across 16 TPU chips...")
        print("#" * 80)

        if is_arm_completed(out_dir, target_tokens=total_tokens):
            print(f">>> [SKIPPING] Arm {run_name} already completed {total_tokens:,} tokens! Skipping.")
            continue

        cleanup_dangling_tpu_processes()
        time.sleep(2)
        sync_codebase_to_workers()

        train_cmd = [
            "scripts/train_distributed.py",
            "--optimizer", opt,
            "--lr", str(lr),
            "--adamw_lr", str(adamw_lr),
            "--seed", str(seed),
            "--total_tokens", str(total_tokens),
            "--batch_size", str(batch_size),
            "--seq_len", str(seq_len),
            "--eval_interval", str(eval_interval),
            "--checkpoint_interval", "1000000",  # Don't save intermediate ckpts for short sweeps
            "--output_dir", str(out_dir),
        ]

        ret = launch_distributed_command(train_cmd)
        if ret != 0:
            print(f"\n[CRITICAL ERROR] Arm {run_name} failed with exit code {ret}!")
            sys.exit(ret)

        cleanup_dangling_tpu_processes()
        sync_runs_back_from_workers()
        time.sleep(2)

        if not is_arm_completed(out_dir, target_tokens=total_tokens):
            print(f"\n[ERROR] Arm {run_name} finished but tokens < {total_tokens:,}!")
            sys.exit(1)

        print(f"\n[ARM SUCCESS] {run_name} finished {total_tokens:,} tokens!")

    total_elapsed_minutes = (time.perf_counter() - t_sweep_start) / 60.0
    print("\n" + "=" * 80)
    print("ALL HYPERPARAMETER SWEEP ARMS COMPLETED SUCCESSFULLY!")
    print(f"Total Sweep Time: {total_elapsed_minutes:.1f} minutes")
    print("=" * 80 + "\n")

    # Run aggregator
    agg_cmd = [sys.executable, "scripts/aggregate_hp_sweep.py"]
    subprocess.run(agg_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Fair Hyperparameter Tuning Sweep on TPU v4-32")
    parser.add_argument("--total_tokens", type=int, default=200_000_000, help="Tokens per arm (default: 200M)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--optimizer", type=str, default=None, choices=["cauchylift", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_hp_sweep(
        total_tokens=args.total_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_interval=args.eval_interval,
        filter_opt=args.optimizer,
        filter_lr=args.lr,
        filter_seed=args.seed,
    )


if __name__ == "__main__":
    main()
