#!/usr/bin/env python3
from __future__ import annotations

import argparse

import torch

from cauchylift import CauchyLift


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend", choices=("cauchylift_hip", "adamw_fused"), default="cauchylift_hip"
    )
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("ROCm device required")
    parameter = torch.nn.Parameter(torch.zeros(4096, 4096, device="cuda"))
    parameter.grad = torch.linspace(-1, 1, parameter.numel(), device="cuda").reshape_as(parameter)
    if args.backend == "cauchylift_hip":
        optimizer = CauchyLift(
            [parameter], lr=1e-7, backend="hip", strict=False
        )
    else:
        optimizer = torch.optim.AdamW(
            [parameter], lr=1e-7, weight_decay=0.0, fused=True
        )
    optimizer.step()
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        optimizer.step()
    torch.cuda.synchronize()
    print(
        {
            "backend": args.backend,
            "iterations": args.iterations,
            "device": torch.cuda.get_device_name(0),
            "hip": torch.version.hip,
        }
    )


if __name__ == "__main__":
    main()
