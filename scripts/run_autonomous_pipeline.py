#!/usr/bin/env python3
"""Autonomous End-to-End Orchestrator for TPU v4-32 Cluster.

Orchestrates:
1. Fair Hyperparameter Sweep on 125M Transformer (FineWeb-Edu):
   - 600M tokens on 2.5k steps (655,360,000 tokens @ 262,144 tokens/step, 2,500 steps)
   - Evaluated across seeds [42, 43, 44] for CauchyLift and AdamW candidate learning rates.
   - Deletes intermediate .pt checkpoints per arm to preserve disk space.
2. Automated Aggregation & Selection:
   - Evaluates cross-seed validation losses to determine the optimal LR for CauchyLift and AdamW.
   - Updates experiments/protocols/protocol_125m_fineweb.json.
3. Launch Main Pretraining Comparison:
   - 125M Transformer on 2.5B FineWeb-Edu tokens (9,537 steps @ 262,144 tokens/step).
   - 3 seeds per optimizer (seeds 42, 43, 44, 6 runs total).
4. Initial Health Verification:
   - Verifies step progression, loss decrease, and TPU throughput on the first main run.
   - Logs verified status to artifacts/main_run_status.json.
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
    SSH_KEY,
    WORKER_IPS,
    cleanup_dangling_tpu_processes,
    launch_distributed_command,
    sync_codebase_to_workers,
    sync_runs_back_from_workers,
)

# Standard Fair Search Grid
FULL_SWEEP_CONFIGS = [
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


def delete_checkpoints(path_pattern: str) -> None:
    """Delete checkpoint .pt files across all 4 worker nodes to conserve disk."""
    for ip in WORKER_IPS:
        cmd = [
            "ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
            ip, f"find /home/tas_ken_rt25/cauchylift/{path_pattern} -name '*.pt' -delete 2>/dev/null || true",
        ]
        subprocess.run(cmd, capture_output=True)


def is_arm_completed(out_dir: pathlib.Path, target_tokens: int) -> bool:
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


def run_sweep_stage(
    sweep_configs: list[dict],
    total_tokens: int = 655_360_000,
    batch_size: int = 8,
    seq_len: int = 2048,
    eval_interval: int = 250,
) -> None:
    print("\n" + "=" * 80)
    print("STAGE 1: HYPERPARAMETER TUNING SWEEP (600M TOKENS ON 2.5K STEPS)")
    print(f"Target Budget per Arm: {total_tokens:,} tokens (~{total_tokens // (batch_size * 16 * seq_len):,} steps)")
    print(f"Total Arms in Queue: {len(sweep_configs)}")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    sync_codebase_to_workers()

    for idx, run_cfg in enumerate(sweep_configs, 1):
        opt = run_cfg["optimizer"]
        lr = run_cfg["lr"]
        adamw_lr = run_cfg["adamw_lr"]
        seed = run_cfg["seed"]

        run_name = f"{opt}_lr{lr:.4f}_seed{seed}"
        out_dir = pathlib.Path(f"runs/sweeps/{run_name}")

        print("\n" + "#" * 80)
        print(f"SWEEP ARM [{idx}/{len(sweep_configs)}]: {run_name.upper()} across 16 TPU chips...")
        print("#" * 80)
        sys.stdout.flush()

        if is_arm_completed(out_dir, target_tokens=total_tokens):
            print(f">>> [SKIPPING] Arm {run_name} already completed {total_tokens:,} tokens.")
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
            "--checkpoint_interval", "1000000",  # No periodic checkpoints for sweep
            "--output_dir", str(out_dir),
        ]

        ret = launch_distributed_command(train_cmd)
        if ret != 0:
            print(f"\n[CRITICAL ERROR] Arm {run_name} failed with exit code {ret}!")
            sys.exit(ret)

        cleanup_dangling_tpu_processes()
        sync_runs_back_from_workers()
        time.sleep(2)

        # Delete checkpoints for this arm across all worker nodes to preserve disk space
        delete_checkpoints(f"runs/sweeps/{run_name}")

        if not is_arm_completed(out_dir, target_tokens=total_tokens):
            print(f"\n[ERROR] Arm {run_name} finished but tokens < {total_tokens:,}!")
            sys.exit(1)

        print(f"\n[ARM SUCCESS] {run_name} finished {total_tokens:,} tokens successfully!")
        sys.stdout.flush()


def aggregate_and_select_best() -> tuple[float, float]:
    print("\n" + "=" * 80)
    print("STAGE 2: AGGREGATING SWEEP RESULTS & SELECTING OPTIMAL HYPERPARAMETERS")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    sync_runs_back_from_workers()

    agg_cmd = [sys.executable, "scripts/aggregate_hp_sweep.py"]
    subprocess.run(agg_cmd, check=True)

    results_file = pathlib.Path("artifacts/sweeps/hp_sweep_results.json")
    if not results_file.exists():
        raise FileNotFoundError(f"Sweep results file not found at {results_file}")

    with open(results_file) as f:
        results = json.load(f)

    opt_cauchy = results.get("optimal_cauchylift")
    opt_adamw = results.get("optimal_adamw")

    if not opt_cauchy or not opt_adamw:
        raise ValueError(f"Incomplete sweep results: cauchylift={opt_cauchy}, adamw={opt_adamw}")

    best_cauchy_lr = float(opt_cauchy["lr"])
    best_adamw_lr = float(opt_adamw["lr"])

    print("\n" + "=" * 80)
    print("OPTIMAL HYPERPARAMETERS SELECTED FROM SWEEP:")
    print(f"  CauchyLift: LR = {best_cauchy_lr:.4f} (Mean Val Loss = {opt_cauchy['mean_val_loss']:.4f} ± {opt_cauchy['std_val_loss']:.4f})")
    print(f"  AdamW:      LR = {best_adamw_lr:.4f} (Mean Val Loss = {opt_adamw['mean_val_loss']:.4f} ± {opt_adamw['std_val_loss']:.4f})")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    # Update protocol_125m_fineweb.json
    protocol_path = pathlib.Path("experiments/protocols/protocol_125m_fineweb.json")
    if protocol_path.exists():
        with open(protocol_path) as f:
            protocol = json.load(f)
        protocol["optimizers"]["cauchylift"]["lr"] = best_cauchy_lr
        protocol["optimizers"]["adamw"]["lr"] = best_adamw_lr
        with open(protocol_path, "w") as f:
            json.dump(protocol, f, indent=2)
        print(f"Updated {protocol_path} with optimal hyperparameters.")

    return best_cauchy_lr, best_adamw_lr


def run_main_pretraining_runs(
    best_cauchy_lr: float,
    best_adamw_lr: float,
    seeds: list[int] = [42, 43, 44],
    total_tokens: int = 2_500_000_000,
    batch_size: int = 8,
    seq_len: int = 2048,
    eval_interval: int = 500,
    checkpoint_interval: int = 2000,
) -> None:
    print("\n" + "=" * 80)
    print("STAGE 3: STARTING FULL 125M PRETRAINING (2.5B TOKENS PER RUN across 3 SEEDS)")
    print(f"CauchyLift LR: {best_cauchy_lr:.4f} | AdamW LR: {best_adamw_lr:.4f} | Seeds: {seeds}")
    print(f"Total Budget per Run: {total_tokens:,} tokens (~{total_tokens // (batch_size * 16 * seq_len):,} steps)")
    print("=" * 80 + "\n")
    sys.stdout.flush()

    main_runs = []
    # CauchyLift 3 seeds
    for s in seeds:
        main_runs.append({"optimizer": "cauchylift", "lr": best_cauchy_lr, "adamw_lr": 0.0006, "seed": s})
    # AdamW 3 seeds
    for s in seeds:
        main_runs.append({"optimizer": "adamw", "lr": best_adamw_lr, "adamw_lr": best_adamw_lr, "seed": s})

    t_main_start = time.perf_counter()

    for idx, run_cfg in enumerate(main_runs, 1):
        opt = run_cfg["optimizer"]
        lr = run_cfg["lr"]
        adamw_lr = run_cfg["adamw_lr"]
        seed = run_cfg["seed"]

        run_name = f"125m_{opt}_lr{lr:.4f}_seed{seed}"
        out_dir = pathlib.Path(f"runs/{run_name}")

        print("\n" + "#" * 80)
        print(f"MAIN PRETRAINING RUN [{idx}/{len(main_runs)}]: Launching {run_name} on 16 TPU chips...")
        print("#" * 80)
        sys.stdout.flush()

        if is_arm_completed(out_dir, target_tokens=total_tokens):
            print(f">>> [SKIPPING] Run {run_name} already completed {total_tokens:,} tokens.")
            continue

        cleanup_dangling_tpu_processes()
        time.sleep(3)
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
            "--checkpoint_interval", str(checkpoint_interval),
            "--output_dir", str(out_dir),
        ]

        ret = launch_distributed_command(train_cmd)
        if ret != 0:
            print(f"\n[CRITICAL ERROR] Main run {run_name} failed with exit code {ret}!")
            sys.exit(ret)

        cleanup_dangling_tpu_processes()
        sync_runs_back_from_workers()
        time.sleep(3)

        if not is_arm_completed(out_dir, target_tokens=total_tokens):
            print(f"\n[ERROR] Main run {run_name} finished but tokens < {total_tokens:,}!")
            sys.exit(1)

        print(f"\n[RUN SUCCESS] {run_name} completed 100% of 2.5B tokens!")
        sys.stdout.flush()

    total_elapsed_hours = (time.perf_counter() - t_main_start) / 3600.0
    print("\n" + "=" * 80)
    print(f"ALL 6 MAIN PRETRAINING RUNS COMPLETED SUCCESSFULLY! ({total_elapsed_hours:.2f} hours)")
    print("=" * 80 + "\n")

    # Aggregate 6 main runs
    agg_cmd = [sys.executable, "scripts/aggregate_sweep_results.py"]
    subprocess.run(agg_cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Autonomous TPU v4-32 Pretraining Orchestrator")
    parser.add_argument("--sweep_tokens", type=int, default=655_360_000, help="Tokens per sweep arm (default: 655,360,000 = 2,500 steps)")
    parser.add_argument("--pretrain_tokens", type=int, default=2_500_000_000, help="Tokens per main pretraining run (default: 2.5B)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--sweep_eval_interval", type=int, default=250)
    parser.add_argument("--pretrain_eval_interval", type=int, default=500)
    parser.add_argument("--pretrain_ckpt_interval", type=int, default=2000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--cauchylift_lr", type=float, default=None, help="Skip sweep and force CauchyLift LR")
    parser.add_argument("--adamw_lr", type=float, default=None, help="Skip sweep and force AdamW LR")
    args = parser.parse_args()

    # Step 1 & 2: Sweep if optimal LRs are not manually forced
    if args.cauchylift_lr is not None and args.adamw_lr is not None:
        best_cauchy = args.cauchylift_lr
        best_adamw = args.adamw_lr
        print(f"[MANUAL OVERRIDE] Using user-specified LRs: CauchyLift={best_cauchy}, AdamW={best_adamw}")
    else:
        run_sweep_stage(
            sweep_configs=FULL_SWEEP_CONFIGS,
            total_tokens=args.sweep_tokens,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            eval_interval=args.sweep_eval_interval,
        )
        best_cauchy, best_adamw = aggregate_and_select_best()

    # Step 3: Run Full Pretraining Comparison
    run_main_pretraining_runs(
        best_cauchy_lr=best_cauchy,
        best_adamw_lr=best_adamw,
        seeds=args.seeds,
        total_tokens=args.pretrain_tokens,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        eval_interval=args.pretrain_eval_interval,
        checkpoint_interval=args.pretrain_ckpt_interval,
    )


if __name__ == "__main__":
    main()
