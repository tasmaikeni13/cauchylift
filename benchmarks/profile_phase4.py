#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import torch

from cauchylift.baselines import create_optimizer
from cauchylift.models import Transformer, TransformerConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile Phase 4 Transformer forward, backward, and optimizer steps.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--optimizer", type=str, default="cauchylift")
    args = parser.parse_args()

    if not torch.cuda.is_available() or not getattr(torch.version, "hip", None):
        raise SystemExit("ROCm device required")

    torch.manual_seed(42)
    device = "cuda"

    # Standard model configuration
    config = TransformerConfig(
        vocab_size=50257,
        hidden_dim=256,
        num_layers=2,
        num_heads=4,
        max_seq_len=128,
        tied_embeddings=True,
        attention_backend="flash",
    )
    model = Transformer(config).to(device)
    optimizer = create_optimizer(args.optimizer, model, lr=1e-3)

    # Synthetic batch
    x = torch.randint(0, 50257, (2, 128), device=device)
    y = torch.randint(0, 50257, (2, 128), device=device)

    # Warmup
    optimizer.zero_grad()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, loss = model(x, y)
    assert loss is not None
    loss.backward()
    optimizer.step()
    torch.cuda.synchronize()

    # Profiled iterations
    for _ in range(args.iterations):
        optimizer.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()

    torch.cuda.synchronize()
    print(
        json.dumps(
            {
                "status": "PROFILED_SUCCESS",
                "iterations": args.iterations,
                "optimizer": args.optimizer,
                "device": torch.cuda.get_device_name(0),
                "hip": torch.version.hip,
                "loss": float(loss.item()),
            }
        )
    )


if __name__ == "__main__":
    main()
