#!/usr/bin/env python3
"""Execute Phase 7 Confirmatory Dual 3B-Token Pretraining (CauchyLift vs. AdamW).

Runs:
1. CauchyLift: 125M Transformer, 3B tokens, lr=0.005, seed 42 on Google Cloud TPU v4-32
2. AdamW:      125M Transformer, 3B tokens, lr=0.0006, seed 42 on Google Cloud TPU v4-32
Then computes side-by-side comparison metrics.
"""

import json
import os
import pathlib
import subprocess
import sys
import time

PYTHON = sys.executable
SCRIPT = str(pathlib.Path(__file__).parent / "train_distributed.py")


def run_command(cmd, log_path):
    print(f"\n[LAUNCHING] {' '.join(cmd)}")
    print(f"[LOG FILE]  {log_path}\n")
    sys.stdout.flush()
    with open(log_path, "a") as log_fp:
        process = subprocess.Popen(
            cmd,
            stdout=log_fp,
            stderr=subprocess.STDOUT,
        )
        process.wait()
        if process.returncode != 0:
            raise RuntimeError(f"Command failed with returncode {process.returncode}")


def main():
    print("=" * 80)
    print("STARTING PHASE 7 CONFIRMATORY 3B-TOKEN PRETRAINING BENCHMARK")
    print("Hardware: Google Cloud TPU v4-32")
    print("Models:   125M Decoder Transformer (FineWeb-Edu, Seed 42)")
    print("=" * 80)

    # 1. CauchyLift Run
    cauchylift_out = pathlib.Path("runs/cauchylift_125m_seed42")
    cauchylift_out.mkdir(parents=True, exist_ok=True)
    cauchylift_cmd = [
        PYTHON, SCRIPT,
        "--optimizer", "cauchylift",
        "--total_tokens", "3000000000",
        "--seq_len", "2048",
        "--batch_size", "4",
        "--lr", "0.005",
        "--momentum", "0.95",
        "--weight_decay", "0.01",
        "--warmup_fraction", "0.10",
        "--seed", "42",
        "--log_interval", "50",
        "--eval_interval", "1000",
        "--checkpoint_interval", "1000",
        "--output_dir", str(cauchylift_out),
    ]

    t0 = time.time()
    run_command(cauchylift_cmd, cauchylift_out / "stdout.log")
    t1 = time.time()
    cauchylift_duration_min = (t1 - t0) / 60.0

    print("\n" + "=" * 80)
    print(f"CauchyLift completed in {cauchylift_duration_min:.2f} minutes!")
    print("Starting AdamW baseline run...")
    print("=" * 80 + "\n")

    # 2. AdamW Run
    adamw_out = pathlib.Path("runs/adamw_125m_seed42")
    adamw_out.mkdir(parents=True, exist_ok=True)
    adamw_cmd = [
        PYTHON, SCRIPT,
        "--optimizer", "adamw",
        "--total_tokens", "3000000000",
        "--seq_len", "2048",
        "--batch_size", "4",
        "--lr", "0.0006",
        "--weight_decay", "0.01",
        "--warmup_fraction", "0.10",
        "--seed", "42",
        "--log_interval", "50",
        "--eval_interval", "1000",
        "--checkpoint_interval", "1000",
        "--output_dir", str(adamw_out),
    ]

    t2 = time.time()
    run_command(adamw_cmd, adamw_out / "stdout.log")
    t3 = time.time()
    adamw_duration_min = (t3 - t2) / 60.0

    print("\n" + "=" * 80)
    print(f"Both runs complete!")
    print(f"CauchyLift: {cauchylift_duration_min:.2f} minutes")
    print(f"AdamW:      {adamw_duration_min:.2f} minutes")
    print("=" * 80)

    # 3. Generate Comparison Summary
    summary_path = pathlib.Path("artifacts/phase7_comparison_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    def parse_metrics(metrics_file):
        evals = []
        train_steps = []
        if metrics_file.exists():
            with open(metrics_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("event") == "eval":
                        evals.append(data)
                    elif "train_loss" in data:
                        train_steps.append(data)
        return train_steps, evals

    cl_train, cl_eval = parse_metrics(cauchylift_out / "metrics.jsonl")
    aw_train, aw_eval = parse_metrics(adamw_out / "metrics.jsonl")

    summary = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cauchylift": {
            "duration_minutes": cauchylift_duration_min,
            "final_train_loss": cl_train[-1]["train_loss"] if cl_train else None,
            "final_val_loss": cl_eval[-1]["val_loss"] if cl_eval else None,
            "final_perplexity": cl_eval[-1]["perplexity"] if cl_eval else None,
            "avg_step_time_ms": sum(s["step_time_ms"] for s in cl_train[-100:]) / max(1, len(cl_train[-100:])) if cl_train else None,
            "avg_tok_per_sec": sum(s["tokens_per_sec"] for s in cl_train[-100:]) / max(1, len(cl_train[-100:])) if cl_train else None,
        },
        "adamw": {
            "duration_minutes": adamw_duration_min,
            "final_train_loss": aw_train[-1]["train_loss"] if aw_train else None,
            "final_val_loss": aw_eval[-1]["val_loss"] if aw_eval else None,
            "final_perplexity": aw_eval[-1]["perplexity"] if aw_eval else None,
            "avg_step_time_ms": sum(s["step_time_ms"] for s in aw_train[-100:]) / max(1, len(aw_train[-100:])) if aw_train else None,
            "avg_tok_per_sec": sum(s["tokens_per_sec"] for s in aw_train[-100:]) / max(1, len(aw_train[-100:])) if aw_train else None,
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SUMMARY WRITTEN] {summary_path}")


if __name__ == "__main__":
    main()
