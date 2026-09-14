#!/usr/bin/env python3
"""Aggregate distributed pretraining sweep results across all 6 runs (125M, 2.5B tokens)."""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import defaultdict


def compute_mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) <= 1:
        return mean, 0.0
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return mean, math.sqrt(variance)


def generate_summary(runs_dir: pathlib.Path = pathlib.Path("runs")):
    print("=" * 80)
    print("AGGREGATING 125M PRETRAINING RESULTS (2.5B TOKENS PER RUN)")
    print("=" * 80)

    # Expected configurations: 3 CauchyLift runs + 3 AdamW runs = 6 runs
    expected_configs = [
        {"optimizer": "cauchylift", "lr": 0.0010, "seeds": [42, 43, 44]},
        {"optimizer": "adamw", "lr": 0.0020, "seeds": [42, 43, 44]},
    ]

    all_run_data = []
    config_groups = defaultdict(list)

    for cfg in expected_configs:
        opt = cfg["optimizer"]
        lr = cfg["lr"]
        group_key = f"{opt.upper()} (LR = {lr:.4f})"

        for seed in cfg["seeds"]:
            run_name = f"125m_{opt}_lr{lr:.4f}_seed{seed}"
            # Also check alternative naming
            candidates = [
                runs_dir / run_name / "run_summary.json",
                runs_dir / f"{opt}_125m_seed{seed}" / "run_summary.json",
                runs_dir / f"125m_{opt}_seed{seed}" / "run_summary.json",
            ]
            found = False
            for c in candidates:
                if c.exists():
                    try:
                        with open(c) as f:
                            data = json.load(f)
                        data["run_dir"] = str(c.parent)
                        all_run_data.append(data)
                        config_groups[group_key].append(data)
                        found = True
                        break
                    except Exception as e:
                        print(f"Warning: Failed reading {c}: {e}")
            if not found:
                print(f"[MISSING] Run summary for {run_name} not found.")

    summary_rows = []
    for group_key, runs in config_groups.items():
        val_losses = [r.get("best_val_loss") for r in runs if r.get("best_val_loss") is not None]
        train_losses = [r.get("final_train_loss") for r in runs if r.get("final_train_loss") is not None]
        tok_speeds = [r.get("tokens_per_sec", r.get("mean_tokens_per_sec", 0.0)) for r in runs]
        mfus = [r.get("mfu", r.get("mean_mfu", 0.0)) for r in runs]
        tokens_seen = [r.get("total_tokens", 0) for r in runs]

        mean_val, std_val = compute_mean_std(val_losses)
        mean_train, std_train = compute_mean_std(train_losses)
        mean_speed, _ = compute_mean_std(tok_speeds)
        mean_mfu, _ = compute_mean_std(mfus)
        completed_seeds = len(val_losses)

        summary_rows.append({
            "config": group_key,
            "completed_seeds": f"{completed_seeds}/3",
            "mean_val_loss": mean_val,
            "std_val_loss": std_val,
            "mean_train_loss": mean_train,
            "std_train_loss": std_train,
            "throughput_tok_s": mean_speed,
            "mfu": mean_mfu,
            "total_tokens_mean": sum(tokens_seen) / max(1, len(tokens_seen)),
        })

    # Rank configurations by Mean Validation Loss (ascending)
    summary_rows.sort(key=lambda x: x["mean_val_loss"] if x["mean_val_loss"] > 0 else float("inf"))

    print("\n" + "=" * 105)
    print(f"{'Rank':<5} | {'Configuration':<26} | {'Seeds':<7} | {'Mean Val Loss':<14} | {'Val Loss Std':<12} | {'Tok/s':<11} | {'MFU (%)':<8}")
    print("-" * 105)
    for rank_idx, row in enumerate(summary_rows, 1):
        print(
            f"{rank_idx:<5} | "
            f"{row['config']:<26} | "
            f"{row['completed_seeds']:<7} | "
            f"{row['mean_val_loss']:<14.4f} | "
            f"{row['std_val_loss']:<12.4f} | "
            f"{row['throughput_tok_s']:<11,.0f} | "
            f"{row['mfu']:<8.1f}%"
        )
    print("=" * 105 + "\n")

    # Save to artifacts/sweep_summary.md and runs/sweep_summary.json
    artifacts_dir = pathlib.Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_md = artifacts_dir / "sweep_summary.md"
    summary_json = runs_dir / "sweep_summary.json"

    with open(summary_json, "w") as f:
        json.dump({
            "ranking": summary_rows,
            "individual_runs": all_run_data,
        }, f, indent=2)

    with open(summary_md, "w") as f:
        f.write("# Distributed Pretraining Sweep Summary: 125M Transformer on 2.5B FineWeb-Edu Tokens\n\n")
        f.write("**Hardware:** 16-Chip Google Cloud TPU v4 Pod Slice (`v4-32`, 32 TensorCores, 4 Hosts, 2x2x4 3D Torus)\n\n")
        f.write("### Configuration Rankings by Mean Validation Loss\n\n")
        f.write("| Rank | Configuration | Completed Seeds | Mean Val Loss | Cross-Seed Std | Mean Train Loss | Throughput (tok/s) | MFU (%) |\n")
        f.write("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for rank_idx, row in enumerate(summary_rows, 1):
            f.write(
                f"| {rank_idx} | **{row['config']}** | {row['completed_seeds']} | "
                f"**{row['mean_val_loss']:.4f}** | ±{row['std_val_loss']:.4f} | "
                f"{row['mean_train_loss']:.4f} | {row['throughput_tok_s']:,.0f} | {row['mfu']:.1f}% |\n"
            )
        f.write("\n### All Individual Runs\n\n")
        f.write("| Run Name | Optimizer | LR | Seed | Tokens Evaluated | Best Val Loss | Final Train Loss | Tok/s | MFU (%) |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in all_run_data:
            f.write(
                f"| `{pathlib.Path(r.get('run_dir', '')).name}` | {r.get('optimizer')} | {r.get('base_lr')} | "
                f"{r.get('seed')} | {r.get('total_tokens', 0):,} | {r.get('best_val_loss', 0.0):.4f} | "
                f"{r.get('final_train_loss', 0.0):.4f} | {r.get('tokens_per_sec', 0.0):,.0f} | {r.get('mfu', 0.0):.1f}% |\n"
            )

    print(f"Saved markdown summary report to: {summary_md}")
    print(f"Saved JSON summary report to:     {summary_json}")


if __name__ == "__main__":
    generate_summary()
