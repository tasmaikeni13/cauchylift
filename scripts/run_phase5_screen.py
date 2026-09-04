from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from dataclasses import asdict
from typing import Any

import torch
import torch.nn as nn

from cauchylift.baselines import create_optimizer
from cauchylift.data import VisionDataset, WikiTextDataset
from cauchylift.models import (
    ConvSSM,
    ConvSSMConfig,
    Transformer,
    TransformerConfig,
    VisionTransformer,
    VisionTransformerConfig,
)
from cauchylift.train.metrics import compute_gradient_and_update_metrics
from cauchylift.train.trainer import TrainingConfig, get_cosine_lr

PROTOCOL_PATH = "experiments/protocols/phase5_protocol.json"
RESULTS_DIR = "artifacts/phase5"
RUNS_DIR = "runs/phase5"


def build_workload(workload_id: str, seed: int, device: str = "cuda"):
    torch.manual_seed(seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(seed)

    if workload_id == "small_decoder_lm":
        config = TransformerConfig(
            vocab_size=50257,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            intermediate_dim=512,
            max_seq_len=256,
            tied_embeddings=True,
            activation="swiglu",
            attention_backend="auto",
        )
        model = Transformer(config).to(device)
        train_ds = WikiTextDataset(split="train", seq_len=256, batch_size=8, seed=seed)
        val_ds = WikiTextDataset(split="valid", seq_len=256, batch_size=8, seed=seed)
        target_losses = [5.50, 5.00, 4.50, 4.00]
        metric_unit = "tokens"

    elif workload_id == "medium_decoder_lm":
        config = TransformerConfig(
            vocab_size=50257,
            hidden_dim=256,
            num_layers=6,
            num_heads=8,
            intermediate_dim=1024,
            max_seq_len=256,
            tied_embeddings=True,
            activation="swiglu",
            attention_backend="auto",
        )
        model = Transformer(config).to(device)
        train_ds = WikiTextDataset(split="train", seq_len=256, batch_size=8, seed=seed)
        val_ds = WikiTextDataset(split="valid", seq_len=256, batch_size=8, seed=seed)
        target_losses = [5.00, 4.50, 4.00, 3.80]
        metric_unit = "tokens"

    elif workload_id == "small_vit":
        config = VisionTransformerConfig(
            img_size=32,
            patch_size=4,
            in_channels=3,
            num_classes=10,
            hidden_dim=128,
            num_layers=4,
            num_heads=4,
            intermediate_dim=512,
            dropout=0.0,
        )
        model = VisionTransformer(config).to(device)
        train_ds = VisionDataset(split="train", batch_size=64, seed=seed)
        val_ds = VisionDataset(split="valid", batch_size=64, seed=seed)
        target_losses = [2.20, 2.00, 1.80, 1.60]
        metric_unit = "examples"

    elif workload_id == "conv_ssm_heldout":
        config = ConvSSMConfig(
            vocab_size=50257,
            hidden_dim=128,
            intermediate_dim=256,
            num_layers=4,
            conv_kernel=7,
            state_dim=16,
            max_seq_len=256,
            tied_embeddings=True,
        )
        model = ConvSSM(config).to(device)
        train_ds = WikiTextDataset(split="test", seq_len=256, batch_size=8, seed=seed)
        val_ds = WikiTextDataset(split="valid", seq_len=256, batch_size=8, seed=seed)
        target_losses = [5.50, 5.00, 4.50, 4.00]
        metric_unit = "tokens"

    else:
        raise ValueError(f"Unknown workload: {workload_id}")

    return model, train_ds, val_ds, target_losses, metric_unit


def evaluate_model(model, val_ds, eval_steps: int = 5, device: str = "cuda") -> float:
    model.eval()
    total_loss = 0.0
    use_bf16 = device.startswith("cuda")
    with torch.no_grad():
        for _ in range(eval_steps):
            x, y, _ = val_ds.next_batch(device=device)
            with torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y)
            total_loss += float(loss.item()) if loss is not None else 0.0
    model.train()
    return total_loss / max(1, eval_steps)


def run_single_experiment(
    run_id: str,
    workload_id: str,
    optimizer_name: str,
    lr: float,
    seed: int,
    max_steps: int = 100,
    eval_interval: int = 10,
    device: str = "cuda",
    custom_opt_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single deterministic training run with rigorous metric tracking."""
    log_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "metrics.jsonl")

    model, train_ds, val_ds, target_losses, metric_unit = build_workload(workload_id, seed, device)

    opt_kwargs = dict(custom_opt_kwargs or {})
    optimizer = create_optimizer(optimizer_name, model, lr=lr, **opt_kwargs)

    train_cfg = TrainingConfig(
        max_steps=max_steps,
        lr=lr,
        min_lr=lr * 0.1,
        warmup_steps=int(max_steps * 0.1),
        device=device,
        precision="bf16",
    )

    records = []
    tokens_or_examples_to_target = {str(t): None for t in target_losses}
    total_progress = 0
    start_time = time.perf_counter()

    opt_times = []
    step_times = []
    loss_history = []
    status = "SUCCESS"
    failure_reason = None

    use_bf16 = device.startswith("cuda")

    with open(log_file, "w") as f_log:
        for step in range(max_steps):
            step_t0 = time.perf_counter()
            optimizer.zero_grad()

            x, y, batch_progress = train_ds.next_batch(device=device)
            total_progress += batch_progress

            with torch.autocast(device_type="cuda" if use_bf16 else "cpu", dtype=torch.bfloat16, enabled=use_bf16):
                _, loss = model(x, y)

            if loss is None or not torch.isfinite(loss):
                status = "FAILED_NAN"
                failure_reason = f"Non-finite loss {loss} at step {step}"
                break

            loss_val = float(loss.item())
            loss_history.append(loss_val)

            # Check loss spike (>4x moving window mean)
            if len(loss_history) > 10:
                recent_mean = sum(loss_history[-10:-1]) / 9.0
                if loss_val > 4.0 * recent_mean and loss_val > 5.0:
                    status = "FAILED_SPIKE"
                    failure_reason = f"Loss spike {loss_val:.2f} > 4x recent mean {recent_mean:.2f} at step {step}"

            loss.backward()

            # Learning rate schedule
            current_lr = get_cosine_lr(step, train_cfg)
            for g in optimizer.param_groups:
                g["lr"] = current_lr

            params_before = {id(p): p.detach().clone() for p in model.parameters() if p.grad is not None}

            # Measure optimizer execution time
            if device.startswith("cuda"):
                torch.cuda.synchronize()
                opt_t0 = time.perf_counter()
                optimizer.step()
                torch.cuda.synchronize()
                opt_time = time.perf_counter() - opt_t0
            else:
                opt_t0 = time.perf_counter()
                optimizer.step()
                opt_time = time.perf_counter() - opt_t0

            step_time = time.perf_counter() - step_t0
            opt_times.append(opt_time)
            step_times.append(step_time)

            # Validation evaluation
            val_loss = None
            if (step + 1) % eval_interval == 0 or step == max_steps - 1:
                val_loss = evaluate_model(model, val_ds, eval_steps=5, device=device)
                for t in target_losses:
                    if val_loss <= t and tokens_or_examples_to_target[str(t)] is None:
                        tokens_or_examples_to_target[str(t)] = total_progress

            # Metrics
            metrics = compute_gradient_and_update_metrics(model, params_before, current_lr)

            record = {
                "step": step + 1,
                "loss": loss_val,
                "val_loss": val_loss,
                "lr": current_lr,
                "progress": total_progress,
                "step_time_sec": step_time,
                "opt_time_sec": opt_time,
                "grad_update_cosine": metrics["grad_update_cosine"],
                "row_concentration": metrics["row_concentration"],
                "col_concentration": metrics["col_concentration"],
                "effective_support": metrics["effective_support"],
                "update_stable_rank": metrics["update_stable_rank"],
                "min_denominator": metrics["min_denominator"],
                "max_denominator": metrics["max_denominator"],
            }
            records.append(record)
            f_log.write(json.dumps(record) + "\n")

            if status != "SUCCESS":
                break

    total_wall_time = time.perf_counter() - start_time
    final_val_loss = records[-1]["val_loss"] if records and records[-1]["val_loss"] is not None else (
        evaluate_model(model, val_ds, eval_steps=5, device=device) if status == "SUCCESS" else None
    )

    summary = {
        "run_id": run_id,
        "workload_id": workload_id,
        "optimizer": optimizer_name,
        "lr": lr,
        "seed": seed,
        "status": status,
        "failure_reason": failure_reason,
        "steps_completed": len(records),
        "total_progress": total_progress,
        "metric_unit": metric_unit,
        "final_loss": loss_history[-1] if loss_history else None,
        "final_val_loss": final_val_loss,
        "best_val_loss": min([r["val_loss"] for r in records if r["val_loss"] is not None], default=None),
        "tokens_to_target": tokens_or_examples_to_target,
        "mean_opt_time_ms": (sum(opt_times) / len(opt_times) * 1000.0) if opt_times else 0.0,
        "mean_step_time_ms": (sum(step_times) / len(step_times) * 1000.0) if step_times else 0.0,
        "total_wall_time_sec": total_wall_time,
    }

    with open(os.path.join(log_dir, "summary.json"), "w") as f_sum:
        json.dump(summary, f_sum, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run Phase 5 Screen and Falsification Experiments")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-tuning", action="store_true", help="Skip tuning if tuning results exist")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    with open(PROTOCOL_PATH) as f:
        protocol = json.load(f)

    workloads = protocol["workloads"]
    optimizers = protocol["optimizers"]
    lr_grid = protocol["hyperparameter_grid"]["lr_grid"]
    tuning_seed = protocol["seeds"]["tuning_seed"]
    conf_seeds = protocol["seeds"]["confirmation_seeds"]

    print("=== CauchyLift Phase 5 Small-Scale Falsification and Screen ===")
    print(f"Device: {args.device}")
    print(f"Workloads: {list(workloads.keys())}")
    print(f"Optimizers: {optimizers}")
    print(f"LR grid: {lr_grid}")

    all_summaries = []

    # -------------------------------------------------------------
    # STAGE 1: TUNING SWEEP ON DECLARED WORKLOADS (W1, W2, W3)
    # -------------------------------------------------------------
    print("\n--- STAGE 1: Hyperparameter Tuning on W1, W2, W3 (Seed 42) ---")
    tuning_workloads = ["small_decoder_lm", "medium_decoder_lm", "small_vit"]
    tuning_results = {w: {opt: {} for opt in optimizers} for w in tuning_workloads}

    for w_id in tuning_workloads:
        for opt in optimizers:
            for lr in lr_grid:
                run_id = f"tune_{w_id}_{opt}_lr{lr}_s{tuning_seed}"
                print(f"Running {run_id} ...", end=" ", flush=True)
                summary = run_single_experiment(
                    run_id=run_id,
                    workload_id=w_id,
                    optimizer_name=opt,
                    lr=lr,
                    seed=tuning_seed,
                    device=args.device,
                )
                tuning_results[w_id][opt][str(lr)] = summary
                all_summaries.append(summary)
                print(f"Status: {summary['status']}, Val Loss: {summary['final_val_loss']}")

    # Select best hyperparameter per optimizer on each workload (or across LM workloads)
    selected_lrs = {w: {} for w in tuning_workloads}
    for w_id in tuning_workloads:
        for opt in optimizers:
            best_lr = None
            best_loss = float("inf")
            for lr in lr_grid:
                res = tuning_results[w_id][opt][str(lr)]
                if res["status"] == "SUCCESS" and res["final_val_loss"] is not None:
                    if res["final_val_loss"] < best_loss:
                        best_loss = res["final_val_loss"]
                        best_lr = lr
            if best_lr is None:
                best_lr = lr_grid[len(lr_grid) // 2]  # Fallback default
            selected_lrs[w_id][opt] = best_lr

    print("\nSelected Hyperparameters from Tuning Stage:")
    for w_id, opt_map in selected_lrs.items():
        print(f"Workload {w_id}: {opt_map}")

    # -------------------------------------------------------------
    # STAGE 2: CONFIRMATORY RUNS ACROSS 3 SEEDS (W1, W2, W3)
    # -------------------------------------------------------------
    print("\n--- STAGE 2: Multi-Seed Confirmation on W1, W2, W3 (Seeds 42, 43, 44) ---")
    conf_results = {w: {opt: [] for opt in optimizers} for w in tuning_workloads}

    for w_id in tuning_workloads:
        for opt in optimizers:
            best_lr = selected_lrs[w_id][opt]
            for seed in conf_seeds:
                # Seed 42 is already run in tuning; we can reuse or run deterministically
                run_id = f"conf_{w_id}_{opt}_lr{best_lr}_s{seed}"
                if seed == tuning_seed:
                    summary = copy.deepcopy(tuning_results[w_id][opt][str(best_lr)])
                    summary["run_id"] = run_id
                else:
                    print(f"Running {run_id} ...", end=" ", flush=True)
                    summary = run_single_experiment(
                        run_id=run_id,
                        workload_id=w_id,
                        optimizer_name=opt,
                        lr=best_lr,
                        seed=seed,
                        device=args.device,
                    )
                    all_summaries.append(summary)
                    print(f"Status: {summary['status']}, Val Loss: {summary['final_val_loss']}")
                conf_results[w_id][opt].append(summary)

    # -------------------------------------------------------------
    # STAGE 3: HELD-OUT WORKLOAD (W4: conv_ssm_heldout)
    # -------------------------------------------------------------
    print("\n--- STAGE 3: Held-Out Workload Evaluation (W4: conv_ssm_heldout, Seeds 42, 43, 44) ---")
    print("Policy: Frozen best LM learning rates used. ZERO hyperparameter tuning permitted on W4!")
    # For each optimizer, select the frozen LR from small_decoder_lm tuning
    heldout_lrs = {opt: selected_lrs["small_decoder_lm"][opt] for opt in optimizers}
    print(f"Frozen held-out LRs: {heldout_lrs}")

    heldout_results = {opt: [] for opt in optimizers}
    for opt in optimizers:
        lr = heldout_lrs[opt]
        for seed in conf_seeds:
            run_id = f"heldout_conv_ssm_{opt}_lr{lr}_s{seed}"
            print(f"Running {run_id} ...", end=" ", flush=True)
            summary = run_single_experiment(
                run_id=run_id,
                workload_id="conv_ssm_heldout",
                optimizer_name=opt,
                lr=lr,
                seed=seed,
                device=args.device,
            )
            heldout_results[opt].append(summary)
            all_summaries.append(summary)
            print(f"Status: {summary['status']}, Val Loss: {summary['final_val_loss']}")

    # -------------------------------------------------------------
    # STAGE 4: ALLOWED ABLATIONS (on small_decoder_lm)
    # -------------------------------------------------------------
    print("\n--- STAGE 4: Predeclared Allowed Ablations on small_decoder_lm ---")
    ablation_results = {}

    best_cauchylift_lr = selected_lrs["small_decoder_lm"]["cauchylift"]

    # 1. Reference PyTorch implementation vs fused HIP kernel
    print("Running reference PyTorch implementation ablation...")
    ref_summary = run_single_experiment(
        run_id=f"ablation_reference_cauchylift_lr{best_cauchylift_lr}_s42",
        workload_id="small_decoder_lm",
        optimizer_name="cauchylift_reference",
        lr=best_cauchylift_lr,
        seed=42,
        device=args.device,
    )
    ablation_results["backend_reference"] = ref_summary
    all_summaries.append(ref_summary)

    # 2. Strict fast HIP vs full status HIP
    print("Running fast HIP path ablation...")
    fast_summary = run_single_experiment(
        run_id=f"ablation_hip_fast_cauchylift_lr{best_cauchylift_lr}_s42",
        workload_id="small_decoder_lm",
        optimizer_name="cauchylift_hip",
        lr=best_cauchylift_lr,
        seed=42,
        device=args.device,
    )
    ablation_results["backend_hip_fast"] = fast_summary
    all_summaries.append(fast_summary)

    # -------------------------------------------------------------
    # SAVE OVERALL SCREEN SUMMARY
    # -------------------------------------------------------------
    final_output = {
        "protocol_hash": "d2228adaee66bdeadffc841aedf989a5edf6084c82d44799990bb5ebf41622ef",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tuning_results": tuning_results,
        "selected_lrs": selected_lrs,
        "confirmation_results": conf_results,
        "heldout_results": heldout_results,
        "ablation_results": ablation_results,
        "total_runs": len(all_summaries),
    }

    summary_path = os.path.join(RESULTS_DIR, "screen_summary.json")
    with open(summary_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nAll experiments complete. Summary written to {summary_path}")


if __name__ == "__main__":
    main()
