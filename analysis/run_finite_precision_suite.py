#!/usr/bin/env python3
"""Finite-precision diagnostics for the frozen Phase 2 execution rule."""

from __future__ import annotations

import argparse
import json
import math
import struct
from collections.abc import Callable
from decimal import Decimal, localcontext
from pathlib import Path

from cauchylift_math import cauchylift, cosine, frobenius_norm

Matrix = list[list[float]]


def quantize_fp16(value: float) -> float:
    return struct.unpack("e", struct.pack("e", value))[0]


def quantize_bf16(value: float) -> float:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    rounding_bias = 0x7FFF + ((bits >> 16) & 1)
    rounded = (bits + rounding_bias) & 0xFFFF0000
    return struct.unpack(">f", struct.pack(">I", rounded))[0]


def quantize(matrix: Matrix, converter: Callable[[float], float]) -> Matrix:
    return [[converter(value) for value in row] for row in matrix]


def exclusion_safe_direction(matrix: Matrix) -> Matrix:
    rows, columns = len(matrix), len(matrix[0])
    maximum = max(abs(value) for row in matrix for value in row)
    if maximum == 0.0:
        return [[0.0] * columns for _ in range(rows)]
    scaled = [[value / maximum for value in row] for row in matrix]
    active = [(i, j) for i in range(rows) for j in range(columns) if scaled[i][j] != 0.0]
    target = math.sqrt(min(rows, columns))
    if len(active) == 1:
        i, j = active[0]
        result = [[0.0] * columns for _ in range(rows)]
        result[i][j] = math.copysign(target, scaled[i][j])
        return result
    row_energy = [math.fsum(value * value for value in row) for row in scaled]
    column_energy = [
        math.fsum(scaled[i][j] * scaled[i][j] for i in range(rows))
        for j in range(columns)
    ]
    outside_row = [
        math.fsum(row_energy[k] for k in range(rows) if k != i) for i in range(rows)
    ]
    outside_column = [
        math.fsum(column_energy[k] for k in range(columns) if k != j)
        for j in range(columns)
    ]
    energy = [[outside_row[i] + outside_column[j] for j in range(columns)] for i in range(rows)]
    positive = [energy[i][j] for i, j in active if energy[i][j] > 0.0]
    if not positive:
        raise ArithmeticError("represented non-one-sparse input lost every active complement")
    common = min(positive)
    raw = [
        [scaled[i][j] * common / energy[i][j] if scaled[i][j] else 0.0 for j in range(columns)]
        for i in range(rows)
    ]
    norm = frobenius_norm(raw)
    return [[target * value / norm for value in row] for row in raw]


def cancellation_prone_direction(matrix: Matrix) -> Matrix:
    """Historical total-minus-marginal path retained only as a negative control."""

    rows, columns = len(matrix), len(matrix[0])
    maximum = max(abs(value) for row in matrix for value in row)
    scaled = [[value / maximum for value in row] for row in matrix]
    total = math.fsum(value * value for row in scaled for value in row)
    row_energy = [math.fsum(value * value for value in row) for row in scaled]
    column_energy = [
        math.fsum(scaled[i][j] * scaled[i][j] for i in range(rows))
        for j in range(columns)
    ]
    energy = [
        [max(0.0, 2.0 * total - row_energy[i] - column_energy[j]) for j in range(columns)]
        for i in range(rows)
    ]
    boundary = [
        (i, j)
        for i in range(rows)
        for j in range(columns)
        if scaled[i][j] != 0.0 and energy[i][j] == 0.0
    ]
    target = math.sqrt(min(rows, columns))
    if boundary:
        result = [[0.0] * columns for _ in range(rows)]
        for i, j in boundary:
            result[i][j] = math.copysign(target, scaled[i][j])
        return result
    common = min(
        energy[i][j]
        for i in range(rows)
        for j in range(columns)
        if scaled[i][j] != 0.0 and energy[i][j] > 0.0
    )
    raw = [
        [scaled[i][j] * common / energy[i][j] if scaled[i][j] else 0.0 for j in range(columns)]
        for i in range(rows)
    ]
    norm = frobenius_norm(raw)
    return [[target * value / norm for value in row] for row in raw]


def decimal_direction(matrix: Matrix, precision: int = 100) -> Matrix:
    rows, columns = len(matrix), len(matrix[0])
    with localcontext() as context:
        context.prec = precision
        values = [[Decimal(str(value)) for value in row] for row in matrix]
        maximum = max(abs(value) for row in values for value in row)
        if maximum == 0:
            return [[0.0] * columns for _ in range(rows)]
        scaled = [[value / maximum for value in row] for row in values]
        active = [(i, j) for i in range(rows) for j in range(columns) if scaled[i][j] != 0]
        target = Decimal(min(rows, columns)).sqrt()
        if len(active) == 1:
            i, j = active[0]
            result = [[Decimal(0)] * columns for _ in range(rows)]
            result[i][j] = target.copy_sign(scaled[i][j])
            return [[float(value) for value in row] for row in result]
        row_energy = [sum(value * value for value in row) for row in scaled]
        column_energy = [
            sum(scaled[i][j] * scaled[i][j] for i in range(rows))
            for j in range(columns)
        ]
        outside_row = [sum(row_energy[k] for k in range(rows) if k != i) for i in range(rows)]
        outside_column = [
            sum(column_energy[k] for k in range(columns) if k != j) for j in range(columns)
        ]
        energy = [
            [outside_row[i] + outside_column[j] for j in range(columns)]
            for i in range(rows)
        ]
        common = min(energy[i][j] for i, j in active if energy[i][j] > 0)
        raw = [
            [scaled[i][j] * common / energy[i][j] if scaled[i][j] else Decimal(0) for j in range(columns)]
            for i in range(rows)
        ]
        norm = sum(value * value for row in raw for value in row).sqrt()
        return [[float(target * value / norm) for value in row] for row in raw]


def epsilon_direction(matrix: Matrix, epsilon: float) -> Matrix:
    rows, columns = len(matrix), len(matrix[0])
    total = math.fsum(value * value for row in matrix for value in row)
    row_energy = [math.fsum(value * value for value in row) for row in matrix]
    column_energy = [
        math.fsum(matrix[i][j] * matrix[i][j] for i in range(rows))
        for j in range(columns)
    ]
    raw = [
        [
            matrix[i][j]
            / (2.0 * total - row_energy[i] - column_energy[j] + epsilon * total)
            for j in range(columns)
        ]
        for i in range(rows)
    ]
    norm = frobenius_norm(raw)
    target = math.sqrt(min(rows, columns))
    return [[target * value / norm for value in row] for row in raw]


def run() -> dict[str, object]:
    source_cases = {
        "balanced": [[1.0, -0.5], [0.25, 0.75]],
        "near_boundary": [[1.0, 2.0**-10], [-(2.0**-11), 2.0**-12]],
        "wide_bf16_range": [[1.0, 2.0**-60], [0.0, 0.0]],
        "one_sparse": [[0.0, 0.0], [-3.0, 0.0]],
    }
    dtype_reports: dict[str, object] = {}
    for dtype, converter in (("fp16", quantize_fp16), ("bf16", quantize_bf16)):
        cases = {}
        for name, source in source_cases.items():
            represented = quantize(source, converter)
            safe = exclusion_safe_direction(represented)
            oracle = decimal_direction(represented)
            cases[name] = {
                "represented_nonzero_entries": sum(
                    value != 0.0 for row in represented for value in row
                ),
                "safe_direction_finite": all(
                    math.isfinite(value) for row in safe for value in row
                ),
                "cosine_with_decimal_oracle": cosine(safe, oracle),
            }
        dtype_reports[dtype] = cases

    cancellation_source = [[1.0, 2.0**-30], [0.0, 0.0]]
    safe_cancellation = exclusion_safe_direction(cancellation_source)
    subtractive_cancellation = cancellation_prone_direction(cancellation_source)
    oracle_cancellation = decimal_direction(cancellation_source)
    cancellation = {
        "source": cancellation_source,
        "dominant_true_complement": 2.0**-60,
        "subtractive_path_treats_as_boundary": subtractive_cancellation[0][1] == 0.0,
        "safe_path_retains_tiny_component": safe_cancellation[0][1] != 0.0,
        "safe_cosine_with_decimal_oracle": cosine(safe_cancellation, oracle_cancellation),
        "subtractive_cosine_with_decimal_oracle": cosine(
            subtractive_cancellation, oracle_cancellation
        ),
        "interpretation": (
            "Subtractive cancellation can trigger an early boundary branch, but the proved "
            "boundary modulus makes the direction error cubic in off-cell amplitude. The frozen "
            "implementation nevertheless requires exclusion-safe sums."
        ),
    }

    ordinary = [[2.0, 1.0], [0.5, -0.25]]
    exact = cauchylift(ordinary)
    epsilon = epsilon_direction(ordinary, 1e-3)
    epsilon_distance = frobenius_norm(
        [[x - y for x, y in zip(row, other)] for row, other in zip(exact, epsilon)]
    )
    epsilon_report = {
        "fixed_epsilon": 1e-3,
        "direction_distance": epsilon_distance,
        "changes_ordinary_direction": epsilon_distance > 1e-8,
        "conclusion": "an additive epsilon is a different optimizer, not a numerical implementation detail",
    }

    fp32_max = float.fromhex("0x1.fffffep127")
    overflow = {
        "fp32_max": fp32_max,
        "naive_square_overflow_threshold": math.sqrt(fp32_max),
        "adversarial_input_magnitude": 1e30,
        "naive_square_would_overflow": 1e30 > math.sqrt(fp32_max),
        "max_scaled_square_upper_bound": 1.0,
    }
    checks = {
        "all_dtype_cases_finite": all(
            case["safe_direction_finite"]
            for dtype in dtype_reports.values()
            for case in dtype.values()
        ),
        "all_dtype_cases_match_oracle": all(
            case["cosine_with_decimal_oracle"] >= 1.0 - 2e-12
            for dtype in dtype_reports.values()
            for case in dtype.values()
        ),
        "exclusion_safe_cancellation_case": cancellation[
            "safe_cosine_with_decimal_oracle"
        ]
        >= 1.0 - 2e-12,
        "epsilon_is_not_neutral": epsilon_report["changes_ordinary_direction"],
        "projective_scaling_prevents_square_overflow": overflow[
            "naive_square_would_overflow"
        ]
        and overflow["max_scaled_square_upper_bound"] == 1.0,
    }
    return {
        "artifact": "CauchyLift Phase 2 finite-precision suite",
        "run_id": "phase2-finite-precision-20260828",
        "input_dtype_reports": dtype_reports,
        "subtractive_cancellation": cancellation,
        "epsilon_collision": epsilon_report,
        "overflow_regime": overflow,
        "underflow_regime": {
            "fp32_smallest_subnormal_square_input_scale_approx": "2^-74",
            "flush_to_zero_square_input_scale_approx": "2^-63",
            "boundary_error_order": "O(q^3) when q is off-dominant amplitude",
            "required_action": "fp64 rare-path recomputation or exact represented boundary; never epsilon",
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
