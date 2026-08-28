#!/usr/bin/env python3
"""Deterministic hostile edge-case audit for the Phase 1 gate.

This is mathematical analysis, not a model-training workload. Expected
counterexamples are emitted alongside positive checks so that later phases
cannot mistake a repaired statement for the statement originally audited.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections.abc import Callable
from pathlib import Path

from cauchylift_math import (
    cauchylift,
    cosine,
    flatten,
    frobenius_norm,
    maximum_absolute_error,
    transpose,
)

Matrix = list[list[float]]


def one_sparse(rows: int, columns: int, i: int, j: int, value: float) -> Matrix:
    return [
        [value if (row, column) == (i, j) else 0.0 for column in range(columns)]
        for row in range(rows)
    ]


def rank(matrix: Matrix, tolerance: float = 1e-12) -> int:
    work = [row[:] for row in matrix]
    rows, columns = len(work), len(work[0])
    result = 0
    scale = max((abs(value) for row in work for value in row), default=0.0)
    threshold = tolerance * max(1.0, scale)
    for column in range(columns):
        if result >= rows:
            break
        pivot = max(range(result, rows), key=lambda row: abs(work[row][column]))
        if abs(work[pivot][column]) <= threshold:
            continue
        work[result], work[pivot] = work[pivot], work[result]
        divisor = work[result][column]
        work[result] = [value / divisor for value in work[result]]
        for row in range(rows):
            if row == result:
                continue
            factor = work[row][column]
            work[row] = [x - factor * y for x, y in zip(work[row], work[result])]
        result += 1
    return result


def quantize_fp16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def quantize_bf16(value: float) -> float:
    """Round a finite binary32 value to bfloat16, returned as a Python float."""

    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", rounded))[0]


def quantize(matrix: Matrix, converter: Callable[[float], float]) -> Matrix:
    return [[converter(value) for value in row] for row in matrix]


def run() -> dict[str, object]:
    checks: dict[str, bool] = {}

    zero_shapes = ((1, 1), (1, 7), (7, 1), (3, 5))
    checks["zero_gradient_is_total"] = all(
        frobenius_norm(cauchylift([[0.0] * columns for _ in range(rows)])) == 0.0
        for rows, columns in zero_shapes
    )

    scalar_positive = cauchylift([[7.0]])
    scalar_negative = cauchylift([[-7.0]])
    checks["scalar_boundary_semantics"] = scalar_positive == [[1.0]] and scalar_negative == [[-1.0]]

    boundary_errors: list[float] = []
    for rows, columns in ((1, 9), (9, 1), (2, 2), (3, 7)):
        i, j = rows - 1, columns - 1
        gradient = one_sparse(rows, columns, i, j, -3.0)
        expected = one_sparse(rows, columns, i, j, -math.sqrt(min(rows, columns)))
        boundary_errors.append(
            maximum_absolute_error(flatten(cauchylift(gradient)), flatten(expected))
        )
    checks["one_sparse_projective_limit"] = max(boundary_errors) == 0.0

    near_boundary: list[dict[str, float]] = []
    boundary_direction = cauchylift([[1.0, 0.0], [0.0, 0.0]])
    for exponent in (1, 2, 4, 8, 12, 40, 160, 300):
        epsilon = 10.0 ** (-exponent)
        gradient = [[1.0, epsilon], [-epsilon, epsilon]]
        direction = cauchylift(gradient)
        near_boundary.append(
            {
                "epsilon": epsilon,
                "direction_error_from_boundary": maximum_absolute_error(
                    flatten(direction), flatten(boundary_direction)
                ),
                "cosine": cosine(gradient, direction),
            }
        )
    checks["near_boundary_is_finite"] = all(
        math.isfinite(value)
        for case in near_boundary
        for key, value in case.items()
        if key != "epsilon"
    )
    checks["near_boundary_converges"] = near_boundary[-1]["direction_error_from_boundary"] == 0.0

    row = [[3.0, -2.0, 0.0, 1.0, -4.0]]
    checks["one_row_one_column_transpose"] = maximum_absolute_error(
        flatten(transpose(cauchylift(row))), flatten(cauchylift(transpose(row)))
    ) <= 1e-14

    repeated = [[1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [1.0, -1.0, 1.0]]
    repeated_direction = cauchylift(repeated)
    checks["repeated_marginal_energies"] = cosine(repeated, repeated_direction) >= 1.0 - 1e-15

    extreme = [[1e308, -1e-308, 0.0], [1e-200, 1e200, -1e100]]
    extreme_direction = cauchylift(extreme)
    checks["extreme_dynamic_range"] = all(math.isfinite(value) for value in flatten(extreme_direction))

    low_precision_source = [[1.0, -0.25, 0.03125], [-0.5, 0.125, -0.015625]]
    oracle = cauchylift(low_precision_source)
    fp16_direction = cauchylift(quantize(low_precision_source, quantize_fp16))
    bf16_direction = cauchylift(quantize(low_precision_source, quantize_bf16))
    low_precision = {
        "fp16_cosine_with_binary64_oracle": cosine(oracle, fp16_direction),
        "bf16_cosine_with_binary64_oracle": cosine(oracle, bf16_direction),
    }
    checks["low_precision_representable_case"] = min(low_precision.values()) >= 1.0 - 1e-12

    rank_deficient = [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 0.0, 0.0]]
    checks["rank_deficient_input_is_total"] = all(
        math.isfinite(value) for value in flatten(cauchylift(rank_deficient))
    )

    # Regression for the original, insufficient generic-rank wording.
    zero_factor_outer = [[1.0, 2.0], [0.0, 0.0]]
    zero_factor_output_rank = rank(cauchylift(zero_factor_outer))
    checks["rank_theorem_zero_factor_counterexample_retained"] = zero_factor_output_rank == 1

    stochastic_distribution = (
        {"gradient": 10.0, "probability": 0.1},
        {"gradient": -0.5, "probability": 0.9},
    )
    stochastic_mean = math.fsum(
        case["gradient"] * case["probability"] for case in stochastic_distribution
    )
    expected_scalar_direction = math.fsum(
        math.copysign(1.0, case["gradient"]) * case["probability"]
        for case in stochastic_distribution
    )
    checks["stochastic_sign_reversal_counterexample"] = (
        stochastic_mean > 0.0 and stochastic_mean * expected_scalar_direction < 0.0
    )

    return {
        "artifact": "CauchyLift Phase 1 adversarial audit",
        "run_id": "phase1-adversarial-20260828",
        "seed": None,
        "cases": {
            "zero_shapes": [list(shape) for shape in zero_shapes],
            "near_one_sparse": near_boundary,
            "one_sparse_maximum_error": max(boundary_errors),
            "low_precision": low_precision,
            "rank_deficient_input_rank": rank(rank_deficient),
        },
        "expected_negative_cases": {
            "old_rank_wording": {
                "input": zero_factor_outer,
                "output_rank": zero_factor_output_rank,
                "consequence": "full generic rank requires every factor entry to be nonzero",
            },
            "unbiased_stochastic_gradient": {
                "distribution": list(stochastic_distribution),
                "mean_gradient": stochastic_mean,
                "expected_cauchylift_direction": expected_scalar_direction,
                "alignment": stochastic_mean * expected_scalar_direction,
                "consequence": "unbiasedness alone does not imply expected descent",
            },
        },
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
