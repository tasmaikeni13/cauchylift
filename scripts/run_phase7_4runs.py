#!/usr/bin/env python3
"""Execute Phase 7 Confirmatory 3B-Token Pretraining (4 Runs Total across 16x TPU v4-32).

Runs:
1. CauchyLift: 125M Transformer, 3B tokens, lr=0.005, Seed 42 on 16x TPU v4-32
2. CauchyLift: 125M Transformer, 3B tokens, lr=0.005, Seed 43 on 16x TPU v4-32
3. AdamW:      125M Transformer, 3B tokens, lr=0.0006, Seed 42 on 16x TPU v4-32
4. AdamW:      125M Transformer, 3B tokens, lr=0.0006, Seed 43 on 16x TPU v4-32
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).parent.resolve()))

from launch_v4_32_distributed import (
    check_ssh_connectivity,
    sync_codebase_to_workers,
    launch_distributed_command,
)

SCRIPT = "scripts/train_distributed.py"


def parse_metrics(metrics_file: pathlib.Path):
    evals = []
    train_steps = []
    if metrics_file.exists():
        with open(metrics_file) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    if data.get("event") == "eval":
                        evals.append(data)
                    elif "train_loss" in data:
                        train_steps.append(data)
                except Exception:
                    continue
    return train_steps, evals


def main():
    print("=" * 80)
    print("STARTING PHASE 7 CONFIRMATORY 4-RUN 3B-TOKEN PRETRAINING BENCHMARK")
    print("Hardware: Google Cloud TPU v4-32 (16 chips, 4 nodes)")
    print("Models:   125M Decoder Transformer on FineWeb-Edu")
    print("Plan:     2 Seeds CauchyLift [42, 43] + 2 Seeds AdamW [42, 43]")
    print("=" * 80)
    sys.stdout.flush()

    if not check_ssh_connectivity():
        print("[ERROR] Cluster connectivity check failed.")
        sys.exit(1)

    sync_codebase_to_workers()

    runs_config = [
        {
            "name": "CauchyLift Seed 42",
            "out_dir": pathlib.Path("runs/cauchylift_125m_seed42"),
            "optimizer": "cauchylift",
            "lr": "0.005",
            "momentum": "0.95",
            "weight_decay": "0.01",
            "seed": "42",
        },
        {
            "name": "CauchyLift Seed 43",
            "out_dir": pathlib.Path("runs/cauchylift_125m_seed43"),
            "optimizer": "cauchylift",
            "lr": "0.005",
            "momentum": "0.95",
            "weight_decay": "0.01",
            "seed": "43",
        },
        {
            "name": "AdamW Seed 42",
            "out_dir": pathlib.Path("runs/adamw_125m_seed42"),
            "optimizer": "adamw",
            "lr": "0.0006",
            "weight_decay": "0.01",
            "seed": "42",
        },
        {
            "name": "AdamW Seed 43",
            "out_dir": pathlib.Path("runs/adamw_125m_seed43"),
            "optimizer": "adamw",
            "lr": "0.0006",
            "weight_decay": "0.01",
            "seed": "43",
        },
    ]

    total_start_time = time.time()
    run_results = {}

    for i, cfg in enumerate(runs_config, start=1):
        print("\n" + "#" * 80)
        print(f"LAUNCHING RUN {i}/{len(runs_config)}: {cfg['name']}")
        print(f"Optimizer: {cfg['optimizer'].upper()} | LR: {cfg['lr']} | Seed: {cfg['seed']}")
        print(f"Output Directory: {cfg['out_dir']}")
        print("#" * 80 + "\n")
        sys.stdout.flush()

        cfg["out_dir"].mkdir(parents=True, exist_ok=True)
        log_file = cfg["out_dir"] / "stdout.log"

        cmd = [
            SCRIPT,
            "--optimizer", cfg["optimizer"],
            "--total_tokens", "3000000000",
            "--seq_len", "2048",
            "--batch_size", "4",
            "--lr", cfg["lr"],
            "--weight_decay", cfg["weight_decay"],
            "--warmup_fraction", "0.10",
            "--seed", cfg["seed"],
            "--log_interval", "50",
            "--eval_interval", "1000",
            "--checkpoint_interval", "1000",
            "--output_dir", str(cfg["out_dir"]),
        ]
        if "momentum" in cfg:
            cmd.extend(["--momentum", cfg["momentum"]])

        t0 = time.time()
        exit_code = launch_distributed_command(cmd)
        t1 = time.time()
        duration_min = (t1 - t0) / 60.0

        if exit_code != 0:
            print(f"[FATAL ERROR] Run {cfg['name']} failed with exit code {exit_code}")
            sys.exit(exit_code)

        print(f"\n[DONE] {cfg['name']} completed in {duration_min:.2f} minutes!")
        sys.stdout.flush()

        train_steps, evals = parse_metrics(cfg["out_dir"] / "metrics.jsonl")
        run_results[cfg["name"]] = {
            "optimizer": cfg["optimizer"],
            "seed": int(cfg["seed"]),
            "duration_minutes": duration_min,
            "final_train_loss": train_steps[-1]["train_loss"] if train_steps else None,
            "final_val_loss": evals[-1]["val_loss"] if evals else None,
            "final_perplexity": evals[-1]["perplexity"] if evals else None,
            "avg_step_time_ms": sum(s["step_time_ms"] for s in train_steps[-100:]) / max(1, len(train_steps[-100:])) if train_steps else None,
            "avg_tok_per_sec": sum(s["tokens_per_sec"] for s in train_steps[-100:]) / max(1, len(train_steps[-100:])) if train_steps else None,
        }

    total_duration_min = (time.time() - total_start_time) / 60.0
    print("\n" + "=" * 80)
    print("ALL 4 RUNS COMPLETED SUCCESSFULLY!")
    print(f"Total Pretraining Benchmark Time: {total_duration_min:.2f} minutes ({total_duration_min / 60.0:.2f} hours)")
    print("=" * 80)

    summary_path = pathlib.Path("artifacts/phase7_comparison_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_duration_minutes": total_duration_min,
            "runs": run_results,
        }, f, indent=2)

    print(f"\n[SUMMARY WRITTEN] {summary_path}")


if __name__ == "__main__":
    main()
