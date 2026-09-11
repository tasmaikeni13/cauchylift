#!/usr/bin/env python3
"""Execute Phase 6 Hyperparameter Pilot Sweep for CauchyLift and AdamW on 125M Transformer."""

import json
import math
import os
import sys
import time
import torch
import torch_xla
import torch_xla.core.xla_model as xm

from cauchylift import CauchyLift
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


def run_pilot_arm(optimizer_name: str, base_lr: float, total_steps: int = 150, batch_size: int = 4, seq_len: int = 2048, seed: int = 42):
    dev = torch_xla.device()
    torch.manual_seed(seed)

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
    model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)

    decay_params = []
    nodecay_params = []
    for p in model.parameters():
        if p.requires_grad:
            if p.ndim >= 2:
                decay_params.append(p)
            else:
                nodecay_params.append(p)

    param_groups = [
        {"params": decay_params, "weight_decay": 0.01},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]

    if optimizer_name == "cauchylift":
        opt = CauchyLift(param_groups, lr=base_lr, momentum=0.95, weight_decay=0.01, backend="auto")
    elif optimizer_name == "adamw":
        opt = torch.optim.AdamW(param_groups, lr=base_lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

    dataset = PackedTokenDataset(split="train", max_seq_len=seq_len, batch_size=batch_size, seed=seed)
    warmup_steps = int(0.10 * total_steps)  # 10% warmup
    min_lr = base_lr * 0.1

    loss_history = []
    initial_loss = None
    step_times = []

    for step in range(1, total_steps + 1):
        lr_now = get_cosine_lr(step, warmup_steps, total_steps, base_lr, min_lr)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        x_cpu, y_cpu, _ = dataset.next_batch()
        x = x_cpu.to(device=dev)
        y = y_cpu.to(device=dev)

        t0 = time.perf_counter()
        opt.zero_grad()
        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)

        assert torch.isfinite(loss), f"Loss became non-finite at step {step}"
        loss_val = float(loss.item())
        if initial_loss is None:
            initial_loss = loss_val

        loss.backward()
        opt.step()
        torch_xla.sync()
        t1 = time.perf_counter()

        step_times.append(t1 - t0)

        if step % 25 == 0 or step == total_steps:
            loss_history.append({"step": step, "loss": loss_val, "lr": lr_now})

    final_loss = loss_history[-1]["loss"]
    loss_reduction = initial_loss - final_loss
    avg_step_ms = (sum(step_times[5:]) / max(1, len(step_times) - 5)) * 1000.0

    return {
        "optimizer": optimizer_name,
        "base_lr": base_lr,
        "total_steps": total_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": loss_reduction,
        "avg_step_ms": avg_step_ms,
        "loss_history": loss_history,
        "diverged": False,
    }


def main():
    print("=" * 75)
    print("Phase 6 Pilot Sweep: 125M Transformer on FineWeb-Edu (Google Cloud TPU v6e)")
    print("=" * 75)

    cauchylift_lrs = [5e-4, 1e-3, 2e-3, 5e-3, 1e-2]
    adamw_lrs = [1e-4, 3e-4, 6e-4, 1e-3, 2e-3]

    results = {"cauchylift": [], "adamw": []}

    print("\n--- Sweeping CauchyLift Learning Rates ---")
    for lr in cauchylift_lrs:
        t0 = time.time()
        res = run_pilot_arm("cauchylift", lr)
        elapsed = time.time() - t0
        print(f"CauchyLift lr={lr:.1e} | init={res['initial_loss']:.4f} -> final={res['final_loss']:.4f} (drop: {res['loss_reduction']:.4f}) | step: {res['avg_step_ms']:.1f}ms | time: {elapsed:.1f}s")
        results["cauchylift"].append(res)

    print("\n--- Sweeping AdamW Learning Rates ---")
    for lr in adamw_lrs:
        t0 = time.time()
        res = run_pilot_arm("adamw", lr)
        elapsed = time.time() - t0
        print(f"AdamW      lr={lr:.1e} | init={res['initial_loss']:.4f} -> final={res['final_loss']:.4f} (drop: {res['loss_reduction']:.4f}) | step: {res['avg_step_ms']:.1f}ms | time: {elapsed:.1f}s")
        results["adamw"].append(res)

    # Find best LR for each
    best_cauchylift = min(results["cauchylift"], key=lambda r: r["final_loss"])
    best_adamw = min(results["adamw"], key=lambda r: r["final_loss"])

    print("\n" + "=" * 75)
    print(f"Optimal CauchyLift LR: {best_cauchylift['base_lr']:.1e} (final loss: {best_cauchylift['final_loss']:.4f})")
    print(f"Optimal AdamW LR:      {best_adamw['base_lr']:.1e} (final loss: {best_adamw['final_loss']:.4f})")
    print("=" * 75)

    out_file = "artifacts/phase6/pilot_sweep_results.json"
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump({
            "workload": "125M Decoder Transformer (FineWeb-Edu)",
            "total_steps_per_arm": 150,
            "tokens_per_arm": 150 * 4 * 2048,
            "best_cauchylift_lr": best_cauchylift["base_lr"],
            "best_adamw_lr": best_adamw["base_lr"],
            "results": results,
        }, f, indent=2)
    print(f"Saved pilot sweep results to {out_file}")


if __name__ == "__main__":
    main()
