#!/usr/bin/env python3
"""Exact two-mode mode-alternation signature for CauchyLift.

For a diagonal two-mode gradient with ratio q, CauchyLift's direction ratio is
q^3. Exact line search therefore forces the next gradient ratio to -q^-3.
The suite checks that identity and contrasts its exponent with control maps.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cauchylift_math import cauchylift, frobenius_norm

Matrix = list[list[float]]


def normalized_gradient(gradient: Matrix) -> Matrix:
    norm = frobenius_norm(gradient)
    return [[value / norm for value in row] for row in gradient]


def sign_diagonal(gradient: Matrix) -> Matrix:
    return [
        [math.copysign(1.0, gradient[i][j]) if gradient[i][j] else 0.0 for j in range(2)]
        for i in range(2)
    ]


def exact_line_next_ratio(
    q: float, lambda_one: float, lambda_two: float, method: str
) -> tuple[float, float, float]:
    gradient = [[q, 0.0], [0.0, 1.0]]
    if method == "cauchylift":
        direction = cauchylift(gradient, radius=1.0)
    elif method == "normalized_gradient":
        direction = normalized_gradient(gradient)
    elif method in ("sign", "row_column_normalization", "exact_polar"):
        direction = sign_diagonal(gradient)
    else:
        raise ValueError(method)
    d1, d2 = direction[0][0], direction[1][1]
    step = (q * d1 + d2) / (lambda_one * d1 * d1 + lambda_two * d2 * d2)
    next_one = q - step * lambda_one * d1
    next_two = 1.0 - step * lambda_two * d2
    return next_one / next_two, d1 / d2, step


def run() -> dict[str, object]:
    lambda_one, lambda_two = 3.0, 5.0
    cases: list[dict[str, object]] = []
    maximum_errors = {
        "cauchylift": 0.0,
        "normalized_gradient": 0.0,
        "sign": 0.0,
        "row_column_normalization": 0.0,
        "exact_polar": 0.0,
    }
    for q in (0.25, 0.5, 2.0, 4.0, 8.0, -2.0, -4.0):
        methods: dict[str, object] = {}
        for method in maximum_errors:
            observed, direction_ratio, step = exact_line_next_ratio(
                q, lambda_one, lambda_two, method
            )
            if method == "cauchylift":
                expected = -1.0 / (q**3)
                expected_direction_ratio = q**3
            elif method == "normalized_gradient":
                expected = -1.0 / q
                expected_direction_ratio = q
            else:
                expected = -math.copysign(1.0, q)
                expected_direction_ratio = math.copysign(1.0, q)
            error = abs(observed - expected)
            maximum_errors[method] = max(maximum_errors[method], error)
            methods[method] = {
                "direction_ratio": direction_ratio,
                "expected_direction_ratio": expected_direction_ratio,
                "next_gradient_ratio": observed,
                "expected_next_gradient_ratio": expected,
                "absolute_error": error,
                "exact_line_step": step,
            }
        cases.append({"initial_gradient_ratio": q, "methods": methods})

    # Applying the exact CauchyLift ratio law twice maps q to q^9. This is an
    # aggressive alternating concentration signature, not monotone balancing.
    alternation = []
    for q in (2.0, 4.0, 8.0):
        first = -1.0 / q**3
        second = -1.0 / first**3
        alternation.append(
            {
                "q0": q,
                "q1": first,
                "q2": second,
                "identity_q2_equals_q0_to_nine": abs(second - q**9) <= 1e-9 * q**9,
            }
        )

    checks = {
        "cauchylift_cubic_recurrence": maximum_errors["cauchylift"] <= 1e-10,
        "normalized_gradient_linear_recurrence": maximum_errors["normalized_gradient"]
        <= 1e-12,
        "sign_control": maximum_errors["sign"] <= 1e-12,
        "row_column_control": maximum_errors["row_column_normalization"] <= 1e-12,
        "polar_control": maximum_errors["exact_polar"] <= 1e-12,
        "two_step_alternation": all(
            case["identity_q2_equals_q0_to_nine"] for case in alternation
        ),
    }
    return {
        "artifact": "CauchyLift Phase 2 exact two-mode mechanism suite",
        "run_id": "phase2-mechanism-20260828",
        "quadratic_curvatures": [lambda_one, lambda_two],
        "cases": cases,
        "maximum_absolute_errors": maximum_errors,
        "two_step_alternation": alternation,
        "falsifiable_signature": (
            "On an isolated diagonal two-mode local model with exact line search, "
            "log|q_next| = -3 log|q| for CauchyLift, versus slope -1 for normalized "
            "gradient and collapse to |q_next|=1 for sign, row/column-normalized, and polar controls."
        ),
        "guardrail": (
            "The second step gives |q_2|=|q_0|^9, so the theorem predicts aggressive "
            "mode alternation, not an unconditional acceleration or balancing result."
        ),
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
