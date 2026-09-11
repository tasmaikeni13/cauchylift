#!/usr/bin/env python3
"""Verify distributed orchestration across 8x Google Cloud TPU v6e chips.

Tests:
1. Multi-core initialization across all 8 TPU ranks.
2. Synchronized forward, loss, and backward pass.
3. Multi-core gradient all-reduction (xm.reduce_gradients).
4. CauchyLift optimizer step on all ranks.
5. Verification of bitwise identical parameter updates and zero inter-rank drift.
"""

import os
import sys
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.runtime as xr

from cauchylift import CauchyLift
from cauchylift.models.transformer import Transformer, TransformerConfig


def _run_rank(index: int):
    dev = xm.xla_device()
    rank = xr.global_ordinal()
    world_size = xr.world_size()

    # Verify world size is 8
    assert world_size == 8, f"Expected world_size 8, got {world_size}"

    # 1. Deterministic model initialization (same seed on all ranks)
    torch.manual_seed(1337)
    cfg = TransformerConfig(
        vocab_size=1024,
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        intermediate_dim=512,
        max_seq_len=64,
        activation="swiglu",
        norm_eps=1e-5,
        tied_embeddings=True,
        attention_backend="flash",
    )
    model = Transformer(cfg).to(device=dev, dtype=torch.bfloat16)

    # 2. Optimizer setup
    optimizer = CauchyLift(
        model.parameters(),
        lr=5e-3,
        momentum=0.95,
        weight_decay=0.01,
        backend="auto",
    )

    max_drift_across_steps = 0.0

    # Run 10 optimization steps
    for step in range(1, 11):
        optimizer.zero_grad()

        # Each rank receives different data (simulating data-parallel distributed training)
        rank_seed = 42 + rank * 1000 + step * 13
        torch.manual_seed(rank_seed)
        x = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len), device=dev)
        y = torch.randint(0, cfg.vocab_size, (4, cfg.max_seq_len), device=dev)

        with torch.autocast(device_type="xla", dtype=torch.bfloat16):
            _, loss = model(x, y)

        loss.backward()

        # All-reduce gradients across all 8 TPU ranks
        xm.reduce_gradients(optimizer)

        # Optimizer step
        optimizer.step()
        torch_xla.sync()

        # Check parameter synchronization across ranks
        step_max_drift = 0.0
        for p in model.parameters():
            if p.requires_grad:
                # Compute mean parameter across all 8 ranks
                p_mean = xm.all_reduce("sum", p.data) / float(world_size)
                drift = (p.data - p_mean).abs().max().item()
                step_max_drift = max(step_max_drift, drift)

        max_drift_across_steps = max(max_drift_across_steps, step_max_drift)

        if rank == 0:
            print(f"[Step {step:2d}/10] Loss: {loss.item():.4f} | Max rank parameter drift: {step_max_drift:.6e}")

    # Final assertion on rank drift
    assert max_drift_across_steps == 0.0, f"Detected non-zero rank drift: {max_drift_across_steps}"

    if rank == 0:
        print("=" * 70)
        print("PASS: 8x TPU v6e distributed orchestration verified with ZERO drift!")
        print("=" * 70)


def main():
    print("Launching 8-way TPU distributed orchestration verification...")
    xmp.spawn(_run_rank, args=())


if __name__ == "__main__":
    main()
