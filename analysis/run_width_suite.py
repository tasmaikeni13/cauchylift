#!/usr/bin/env python3
"""Shape-totality and frozen-radius width-transfer checks for Phase 2."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cauchylift_math import cauchylift, frobenius_norm


def radius(rows: int, columns: int) -> float:
    return math.sqrt(min(rows, columns))


def shape_report(rows: int, columns: int) -> dict[str, float | int | bool]:
    squared = min(rows, columns)
    return {
        "rows": rows,
        "columns": columns,
        "radius": math.sqrt(squared),
        "average_squared_row_norm": squared / rows,
        "average_squared_column_norm": squared / columns,
        "row_fiber_cap_holds": squared / rows <= 1.0,
        "column_fiber_cap_holds": squared / columns <= 1.0,
        "one_fiber_cap_is_tight": squared == rows or squared == columns,
        "transpose_radius_equal": radius(rows, columns) == radius(columns, rows),
    }


def run(spec_path: Path) -> dict[str, object]:
    specification = json.loads(spec_path.read_text(encoding="utf-8"))
    width_families: dict[str, list[dict[str, float | int | bool]]] = {
        "square": [],
        "expansion_4x": [],
        "contraction_4x": [],
        "vocabulary_table": [],
        "vector": [],
    }
    for width in (64, 128, 256, 512, 1024):
        width_families["square"].append(shape_report(width, width))
        width_families["expansion_4x"].append(shape_report(4 * width, width))
        width_families["contraction_4x"].append(shape_report(width, 4 * width))
        width_families["vocabulary_table"].append(shape_report(50304, width))
        width_families["vector"].append(shape_report(width, 1))

    decoder_shapes = {
        "token_embedding": [50304, 768],
        "attention_qkv": [2304, 768],
        "attention_output": [768, 768],
        "mlp_up": [3072, 768],
        "mlp_down": [768, 3072],
        "normalization_gain": [768, 1],
        "scalar": [1, 1],
    }
    decoder_report = {
        name: shape_report(rows, columns)
        for name, (rows, columns) in decoder_shapes.items()
    }

    small_totality = []
    for rows, columns in ((1, 1), (1, 7), (7, 1), (2, 9), (9, 2)):
        zero = [[0.0] * columns for _ in range(rows)]
        one_sparse = [[0.0] * columns for _ in range(rows)]
        one_sparse[-1][-1] = -2.0
        zero_direction = cauchylift(zero)
        boundary_direction = cauchylift(one_sparse)
        small_totality.append(
            {
                "shape": [rows, columns],
                "zero_maps_to_zero": frobenius_norm(zero_direction) == 0.0,
                "one_sparse_radius": frobenius_norm(boundary_direction),
                "expected_radius": radius(rows, columns),
                "one_sparse_sign_preserved": boundary_direction[-1][-1] < 0.0,
            }
        )

    all_shape_reports = [
        report for family in width_families.values() for report in family
    ] + list(decoder_report.values())
    checks = {
        "radius_matches_spec": specification["mathematical_map"]["radius"]
        == "rho(m,n) = sqrt(min(m,n))",
        "both_average_fiber_caps": all(
            report["row_fiber_cap_holds"] and report["column_fiber_cap_holds"]
            for report in all_shape_reports
        ),
        "maximal_symmetric_radius": all(
            report["one_fiber_cap_is_tight"] for report in all_shape_reports
        ),
        "transpose_radius": all(
            report["transpose_radius_equal"] for report in all_shape_reports
        ),
        "zero_and_boundary_totality": all(
            case["zero_maps_to_zero"]
            and abs(case["one_sparse_radius"] - case["expected_radius"]) <= 1e-14
            and case["one_sparse_sign_preserved"]
            for case in small_totality
        ),
        "no_fallback": specification["initial_decoder_contract"]["fallback_optimizer"]
        is None,
        "all_parameter_kinds_declared": len(
            specification["initial_decoder_contract"]["parameter_groups"]
        )
        == 5,
    }
    return {
        "artifact": "CauchyLift Phase 2 width and shape suite",
        "run_id": "phase2-width-20260828",
        "specification": str(spec_path),
        "radius_derivation": (
            "rho^2 <= m and rho^2 <= n are exactly the requirements that average "
            "squared row and column update norms are both at most one. The maximal "
            "transpose-symmetric choice is rho^2=min(m,n)."
        ),
        "width_transfer_families": width_families,
        "decoder_parameter_shapes": decoder_report,
        "small_shape_totality": small_totality,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec", type=Path, default=Path("spec/optimizer_v0.2.json")
    )
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.spec)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
