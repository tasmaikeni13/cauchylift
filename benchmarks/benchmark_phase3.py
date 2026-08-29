#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import pathlib
import statistics
import time
from collections.abc import Callable
from typing import Any

import torch

from cauchylift import CauchyLift


SEED = 20260828
REPRESENTATIVE_SHAPES = (
    [(50304, 768)]
    + sum(
        (
            [(2304, 768), (768, 768), (3072, 768), (768, 3072), (768,), (768,)]
            for _ in range(12)
        ),
        [],
    )
)
SWEEP_SHAPES = [(64, 64), (256, 256), (1024, 1024), (4096, 4096), (32000, 768)]


def summary(samples: list[float]) -> dict[str, Any]:
    ordered = sorted(samples)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    return {
        "samples_ms": samples,
        "median_ms": median,
        "mad_ms": statistics.median(deviations),
        "minimum_ms": ordered[0],
        "maximum_ms": ordered[-1],
        "p25_ms": ordered[len(ordered) // 4],
        "p75_ms": ordered[(3 * len(ordered)) // 4],
    }


def timed(function: Callable[[], None], warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def parameter_bytes(parameters: list[torch.Tensor]) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in parameters)


def native_logical_bytes(shapes: list[tuple[int, ...]], element_size: int) -> int:
    """Count declared global-memory operations in the prevalidated HIP graph."""
    total = 0
    for shape in shapes:
        rows = shape[0] if shape else 1
        columns = math.prod(shape[1:]) if len(shape) > 1 else 1
        elements = rows * columns
        tile_rows = (rows + 63) // 64
        tile_columns = (columns + 63) // 64
        tiles = tile_rows * tile_columns
        # Three gradient reads, one parameter read/write, and row/column/total
        # energy reads in both the norm and output passes.
        total += elements * (5 * element_size + 24)
        # FP32 marginal/total/norm atomic read-modify-writes and initialization.
        total += 8 * (rows * tile_columns + columns * tile_rows + 2 * tiles)
        total += 4 * (rows + columns)
    return total


def state_summary(optimizer: torch.optim.Optimizer) -> dict[str, int]:
    tensors = [
        value
        for state in optimizer.state.values()
        for value in state.values()
        if torch.is_tensor(value)
    ]
    return {
        "tensor_count": len(tensors),
        "bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
    }


def make_parameters(shapes: list[tuple[int, ...]], dtype: torch.dtype) -> list[torch.nn.Parameter]:
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    parameters = []
    for shape in shapes:
        parameter = torch.nn.Parameter(torch.zeros(shape, device="cuda", dtype=dtype))
        parameter.grad = torch.randn(shape, generator=generator, device="cuda", dtype=dtype)
        parameters.append(parameter)
    return parameters


def clean() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def measure_case(
    name: str,
    shapes: list[tuple[int, ...]],
    dtype: torch.dtype,
    warmup: int,
    iterations: int,
) -> dict[str, Any]:
    parameters = make_parameters(shapes, dtype)
    gradients = [parameter.grad for parameter in parameters]
    fresh = [gradient.clone() for gradient in gradients]
    parameter_size = parameter_bytes(parameters)
    element_count = sum(parameter.numel() for parameter in parameters)
    learning_rate = 1e-7

    if name == "cauchylift_hip":
        for gradient in gradients:
            if not bool(torch.isfinite(gradient).all()) or int(torch.count_nonzero(gradient)) < 2:
                raise RuntimeError("fast HIP benchmark requires finite non-boundary gradients")
            rows = gradient.shape[0] if gradient.ndim else 1
            matrix = gradient.float().reshape(rows, -1)
            squares = matrix.square()
            active_squares = squares[matrix != 0]
            if not bool(torch.isfinite(active_squares).all()) or bool((active_squares == 0).any()):
                raise RuntimeError("fast HIP benchmark requires representable active FP32 squares")
            total = squares.sum()
            denominator_lower_bound = (
                total - squares.sum(dim=1).max()
                + total
                - squares.sum(dim=0).max()
            )
            if not bool(torch.isfinite(total)) or not bool(denominator_lower_bound > 0):
                raise RuntimeError("fast HIP benchmark requires positive finite FP32 denominators")
        del active_squares, denominator_lower_bound, matrix, squares, total
        optimizer = CauchyLift(
            parameters, lr=learning_rate, backend="hip", strict=False
        )
        step = optimizer.step
        logical_bytes = native_logical_bytes(shapes, parameters[0].element_size())
    elif name == "cauchylift_reference":
        optimizer = CauchyLift(parameters, lr=learning_rate, backend="reference")
        step = optimizer.step
        logical_bytes = 56 * element_count
    elif name == "adamw_fused":
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=0.0, fused=True
        )
        step = optimizer.step
        logical_bytes = 7 * parameter_size
    elif name == "adamw_foreach":
        optimizer = torch.optim.AdamW(
            parameters, lr=learning_rate, weight_decay=0.0, foreach=True
        )
        step = optimizer.step
        logical_bytes = 7 * parameter_size
    elif name == "copy_lower_bound":
        optimizer = None
        @torch.no_grad()
        def step() -> None:
            torch._foreach_add_(parameters, gradients, alpha=-learning_rate)

        logical_bytes = 3 * parameter_size
    else:
        raise ValueError(name)

    # Allocate AdamW state and JIT-load CauchyLift before measuring.
    step()
    torch.cuda.synchronize()
    persistent = state_summary(optimizer) if optimizer is not None else {"tensor_count": 0, "bytes": 0}
    optimizer_samples = timed(step, warmup, iterations)

    def complete() -> None:
        torch._foreach_copy_(gradients, fresh)
        step()

    complete_samples = timed(complete, warmup, iterations)
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    transient = max(0, torch.cuda.max_memory_allocated() - baseline)
    result = {
        "name": name,
        "shape_count": len(shapes),
        "element_count": element_count,
        "parameter_bytes": parameter_size,
        "estimated_logical_bytes_per_optimizer_step": logical_bytes,
        "optimizer_only": summary(optimizer_samples),
        "complete_parameter_update": summary(complete_samples),
        "achieved_logical_bandwidth_gbps": logical_bytes
        / (statistics.median(optimizer_samples) * 1e6),
        "peak_transient_bytes": transient,
        "persistent_optimizer": persistent,
        "prevalidated_finite_input": name == "cauchylift_hip",
        "strict_host_status_check": False if name == "cauchylift_hip" else None,
    }
    del fresh, gradients, parameters, optimizer
    clean()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or not torch.version.hip:
        raise SystemExit("ROCm device required")

    torch.manual_seed(SEED)
    torch.cuda.set_device(0)
    started = time.time()
    representative = []
    for name in (
        "copy_lower_bound",
        "adamw_fused",
        "adamw_foreach",
        "cauchylift_reference",
        "cauchylift_hip",
    ):
        representative.append(
            measure_case(
                name,
                list(REPRESENTATIVE_SHAPES),
                torch.bfloat16,
                args.warmup,
                args.iterations,
            )
        )

    sweep = []
    for shape in SWEEP_SHAPES:
        cases = []
        for name in ("copy_lower_bound", "adamw_fused", "cauchylift_hip"):
            cases.append(
                measure_case(
                    name,
                    [shape],
                    torch.float32,
                    args.warmup,
                    args.iterations,
                )
            )
        sweep.append({"shape": list(shape), "cases": cases})

    by_name = {case["name"]: case for case in representative}
    best_adamw = min(
        (by_name["adamw_fused"], by_name["adamw_foreach"]),
        key=lambda value: value["optimizer_only"]["median_ms"],
    )
    ratio = (
        by_name["cauchylift_hip"]["optimizer_only"]["median_ms"]
        / best_adamw["optimizer_only"]["median_ms"]
    )
    result = {
        "schema_version": 1,
        "seed": SEED,
        "environment": {
            "torch": torch.__version__,
            "hip_runtime": torch.version.hip,
            "device": torch.cuda.get_device_name(0),
            "device_capability": list(torch.cuda.get_device_capability(0)),
        },
        "timing_protocol": {
            "clock": "torch.cuda.Event on the current ROCm stream",
            "synchronization": "end event synchronized for every sample",
            "warmup": args.warmup,
            "iterations": args.iterations,
            "dispersion": "MAD and sampled p25/p75",
            "representative_dtype": "bfloat16",
            "matrix_sweep_dtype": "float32",
            "learning_rate": 1e-7,
        },
        "representative_shapes": [list(shape) for shape in REPRESENTATIVE_SHAPES],
        "representative_suite": representative,
        "matrix_size_sweep": sweep,
        "gate": {
            "best_adamw_path": best_adamw["name"],
            "cauchylift_to_best_adamw_ratio": ratio,
            "within_15_percent": ratio <= 1.15,
        },
        "elapsed_seconds": time.time() - started,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    result["payload_sha256_before_embedding"] = hashlib.sha256(payload.encode()).hexdigest()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
