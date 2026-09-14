#!/usr/bin/env python3
"""Master Pretraining Sweep Runner for 125M Transformer on 16-Chip TPU v4-32 Pod Slice.

Executes the exhaustive 12-run distributed pretraining sweep (3B tokens per run):
- CauchyLift LR=0.0025 across seeds [42, 43, 44] (3 runs)
- CauchyLift LR=0.0050 across seeds [42, 43, 44] (3 runs)
- CauchyLift LR=0.0100 across seeds [42, 43, 44] (3 runs)
- AdamW Baseline LR=0.0006 across seeds [42, 43, 44] (3 runs)

Total: 12 complete pretraining runs (36 Billion total tokens trained).
Zero simulated runs, zero early breaks, full TPU memory cleanup and multi-host synchronization between runs.
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


SWEEP_CONFIGS = [
    # CauchyLift Grid
    {"optimizer": "cauchylift", "lr": 0.0025, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "cauchylift", "lr": 0.0025, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "cauchylift", "lr": 0.0025, "adamw_lr": 0.0006, "seed": 44},

    {"optimizer": "cauchylift", "lr": 0.0050, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "cauchylift", "lr": 0.0050, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "cauchylift", "lr": 0.0050, "adamw_lr": 0.0006, "seed": 44},

    {"optimizer": "cauchylift", "lr": 0.0100, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "cauchylift", "lr": 0.0100, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "cauchylift", "lr": 0.0100, "adamw_lr": 0.0006, "seed": 44},

    # AdamW Baseline
    {"optimizer": "adamw", "lr": 0.0006, "adamw_lr": 0.0006, "seed": 42},
    {"optimizer": "adamw", "lr": 0.0006, "adamw_lr": 0.0006, "seed": 43},
    {"optimizer": "adamw", "lr": 0.0006, "adamw_lr": 0.0006, "seed": 44},
]


def is_run_completed(out_dir: pathlib.Path, target_tokens: int = 3_000_000_000) -> bool:
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



def run_sweep(
    total_tokens: int = 3_000_000_000,
    batch_size: int = 8,
    seq_len: int = 2048,
    eval_interval: int = 500,
    checkpoint_interval: int = 2000,
    filter_opt: str | None = None,
    filter_lr: float | None = None,
    filter_seed: int | None = None,
):
    print("\n" + "=" * 80)
    print("STARTING 125M PRETRAINING SWEEP (12 RUNS x 3B TOKENS)")
    print("Google Cloud TPU v4-32 (16 Chips, 4 Worker Nodes)")
    print(f"Total Target Tokens per Run: {total_tokens:,}")
    print(f"Global Batch: {batch_size * 16 * seq_len:,} tokens/step")
    print("=" * 80 + "\n")

    sync_codebase_to_workers()

    selected_runs = []
    for cfg in SWEEP_CONFIGS:
        if filter_opt and cfg["optimizer"] != filter_opt:
            continue
        if filter_lr and abs(cfg["lr"] - filter_lr) > 1e-6:
            continue
        if filter_seed and cfg["seed"] != filter_seed:
            continue
        selected_runs.append(cfg)

    print(f"Total Runs in Queue: {len(selected_runs)}")
    for idx, r in enumerate(selected_runs, 1):
        print(f"  [{idx:2d}/{len(selected_runs)}] {r['optimizer'].upper()} | LR: {r['lr']:.4f} | Seed: {r['seed']}")
    print("-" * 80 + "\n")

    t_sweep_start = time.perf_counter()

    for idx, run_cfg in enumerate(selected_runs, 1):
        opt = run_cfg["optimizer"]
        lr = run_cfg["lr"]
        adamw_lr = run_cfg["adamw_lr"]
        seed = run_cfg["seed"]

        run_name = f"125m_{opt}_lr{lr:.4f}_seed{seed}"
        out_dir = pathlib.Path(f"runs/{run_name}")

        print("\n" + "#" * 80)
        print(f"QUEUE [{idx}/{len(selected_runs)}]: Launching {run_name} across all 16 TPU chips...")
        print("#" * 80)

        if is_run_completed(out_dir, target_tokens=total_tokens):
            print(f">>> [SKIPPING] Run {run_name} already completed 3B tokens! Skipping.")
            continue

        # Step 1: Clean up any dangling processes on TPU hardware
        cleanup_dangling_tpu_processes()
        time.sleep(3)

        # Step 2: Ensure latest code is synchronized
        sync_codebase_to_workers()

        # Step 3: Launch distributed pretraining command
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
            "--checkpoint_interval", str(checkpoint_interval),
            "--output_dir", str(out_dir),
        ]

        ret = launch_distributed_command(train_cmd)
        if ret != 0:
            print(f"\n[CRITICAL ERROR] Run {run_name} failed with exit code {ret}!")
            sys.exit(ret)

        # Step 4: Cleanup TPU after run and sync artifacts
        cleanup_dangling_tpu_processes()
        sync_runs_back_from_workers()
        time.sleep(3)

        # Verify completion
        if not is_run_completed(out_dir, target_tokens=total_tokens):
            print(f"\n[ERROR] Run {run_name} finished but tokens < {total_tokens:,}!")
            sys.exit(1)

        print(f"\n[RUN SUCCESS] {run_name} finished 100% of target steps!")

    # Step 5: Aggregate all results
    total_elapsed_hours = (time.perf_counter() - t_sweep_start) / 3600.0
    print("\n" + "=" * 80)
    print("ALL SCHEDULED PRETRAINING RUNS COMPLETED SUCCESSFULLY!")
    print(f"Total Sweep Elapsed Time: {total_elapsed_hours:.2f} hours")
    print("=" * 80 + "\n")

    # Run aggregator
    agg_cmd = [sys.executable, "scripts/aggregate_sweep_results.py"]
    subprocess.run(agg_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Full 125M Hyperparameter Sweep Runner on TPU v4-32")
    parser.add_argument("--total_tokens", type=int, default=3_000_000_000, help="Total tokens per run (default: 3B)")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per chip (default: 8)")
    parser.add_argument("--seq_len", type=int, default=2048, help="Context length (default: 2048)")
    parser.add_argument("--eval_interval", type=int, default=500, help="Validation interval (default: 500)")
    parser.add_argument("--optimizer", type=str, default=None, choices=["cauchylift", "adamw"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    run_sweep(
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
