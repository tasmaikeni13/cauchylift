#!/usr/bin/env python3
"""Deterministic finite-distribution checks for the Phase 2 noise theory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cauchylift_math import cauchylift, frobenius_norm, inner

Matrix = list[list[float]]
Distribution = list[tuple[float, Matrix]]


def add(left: Matrix, right: Matrix) -> Matrix:
    return [[x + y for x, y in zip(row, other)] for row, other in zip(left, right)]


def subtract(left: Matrix, right: Matrix) -> Matrix:
    return [[x - y for x, y in zip(row, other)] for row, other in zip(left, right)]


def scale(matrix: Matrix, factor: float) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def expectation(distribution: Distribution) -> Matrix:
    rows, columns = len(distribution[0][1]), len(distribution[0][1][0])
    return [
        [
            math.fsum(probability * matrix[i][j] for probability, matrix in distribution)
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def exact_distribution_report(distribution: Distribution) -> dict[str, float | bool]:
    mu = expectation(distribution)
    rows, columns = len(mu), len(mu[0])
    rho = math.sqrt(min(rows, columns))
    gamma = 1.0 / math.sqrt(3.0)
    expected_sample_norm = math.fsum(
        probability * frobenius_norm(matrix) for probability, matrix in distribution
    )
    expected_noise_norm = math.fsum(
        probability * frobenius_norm(subtract(matrix, mu))
        for probability, matrix in distribution
    )
    noise_second_moment = math.fsum(
        probability * frobenius_norm(subtract(matrix, mu)) ** 2
        for probability, matrix in distribution
    )
    sigma = math.sqrt(noise_second_moment)
    expected_alignment = math.fsum(
        probability * inner(mu, cauchylift(matrix)) for probability, matrix in distribution
    )
    first_moment_lower = rho * (
        gamma * expected_sample_norm - expected_noise_norm
    )
    second_moment_lower = rho * (gamma * frobenius_norm(mu) - sigma)
    return {
        "rho": rho,
        "true_gradient_norm": frobenius_norm(mu),
        "expected_sample_norm": expected_sample_norm,
        "expected_noise_norm": expected_noise_norm,
        "noise_rms_sigma": sigma,
        "expected_alignment": expected_alignment,
        "first_moment_lower_bound": first_moment_lower,
        "second_moment_lower_bound": second_moment_lower,
        "first_moment_condition_holds": expected_noise_norm
        < gamma * expected_sample_norm,
        "signal_to_noise_condition_holds": sigma < gamma * frobenius_norm(mu),
        "bounds_hold": expected_alignment + 1e-12
        >= max(first_moment_lower, second_moment_lower),
    }


def benign_distribution() -> Distribution:
    mu = [[2.0, 0.5], [0.25, 1.0]]
    perturbations = (
        [[0.12, 0.0], [0.0, -0.06]],
        [[0.0, 0.08], [-0.04, 0.0]],
        [[0.03, -0.05], [0.02, 0.01]],
        [[-0.02, 0.01], [0.0, 0.04]],
    )
    distribution: Distribution = []
    for perturbation in perturbations:
        distribution.append((0.125, add(mu, perturbation)))
        distribution.append((0.125, subtract(mu, perturbation)))
    return distribution


def interior_bias_report(distribution: Distribution) -> dict[str, float | bool]:
    mu = expectation(distribution)
    expected_direction = expectation(
        [(probability, cauchylift(matrix)) for probability, matrix in distribution]
    )
    deterministic_direction = cauchylift(mu)
    observed_bias = frobenius_norm(subtract(expected_direction, deterministic_direction))

    normalized_samples = []
    minimum_h = 2.0
    for probability, matrix in distribution:
        norm = frobenius_norm(matrix)
        unit = scale(matrix, 1.0 / norm)
        normalized_samples.append((probability, unit))
        row = [math.fsum(value * value for value in values) for values in unit]
        column = [
            math.fsum(unit[i][j] * unit[i][j] for i in range(len(unit)))
            for j in range(len(unit[0]))
        ]
        minimum_h = min(
            minimum_h,
            min(2.0 - row[i] - column[j] for i in range(len(unit)) for j in range(len(unit[0]))),
        )
    mu_unit = scale(mu, 1.0 / frobenius_norm(mu))
    expected_projective_input_error = math.fsum(
        probability * frobenius_norm(subtract(unit, mu_unit))
        for probability, unit in normalized_samples
    )
    rho = math.sqrt(min(len(mu), len(mu[0])))
    lipschitz = 4.0 / minimum_h + 16.0 / (minimum_h * minimum_h)
    upper_bound = rho * lipschitz * expected_projective_input_error
    return {
        "minimum_normalized_denominator": minimum_h,
        "observed_bias_norm": observed_bias,
        "lipschitz_bias_upper_bound": upper_bound,
        "bound_holds": observed_bias <= upper_bound + 1e-12,
    }


def smooth_descent_report(distribution: Distribution) -> dict[str, float | bool]:
    point = expectation(distribution)
    report = exact_distribution_report(distribution)
    rho = float(report["rho"])
    margin = float(report["second_moment_lower_bound"]) / rho
    step = margin / rho  # strictly below the 2*margin/rho threshold for L=1
    initial_value = 0.5 * frobenius_norm(point) ** 2
    expected_next_value = math.fsum(
        probability
        * 0.5
        * frobenius_norm(subtract(point, scale(cauchylift(matrix), step))) ** 2
        for probability, matrix in distribution
    )
    proved_upper = initial_value - step * float(report["expected_alignment"]) + 0.5 * step**2 * rho**2
    return {
        "objective": "f(W)=0.5*||W||_F^2",
        "step": step,
        "initial_value": initial_value,
        "expected_next_value": expected_next_value,
        "smoothness_upper_bound": proved_upper,
        "strict_descent": expected_next_value < initial_value,
        "upper_bound_holds": expected_next_value <= proved_upper + 1e-12,
    }


def adversarial_distribution() -> Distribution:
    return [(0.1, [[10.0]]), (0.9, [[-0.5]])]


def run() -> dict[str, object]:
    benign = benign_distribution()
    benign_report = exact_distribution_report(benign)
    bias = interior_bias_report(benign)
    descent = smooth_descent_report(benign)
    adverse = adversarial_distribution()
    adverse_report = exact_distribution_report(adverse)
    adverse_mu = expectation(adverse)
    adverse_expected_direction = expectation(
        [(probability, cauchylift(matrix)) for probability, matrix in adverse]
    )
    adverse_alignment = inner(adverse_mu, adverse_expected_direction)
    checks = {
        "benign_first_moment_condition": bool(benign_report["first_moment_condition_holds"]),
        "benign_snr_condition": bool(benign_report["signal_to_noise_condition_holds"]),
        "alignment_bounds": bool(benign_report["bounds_hold"]),
        "interior_bias_bound": bool(bias["bound_holds"]),
        "one_step_smooth_descent": bool(descent["strict_descent"] and descent["upper_bound_holds"]),
        "unbiased_noise_counterexample": adverse_mu[0][0] > 0.0 and adverse_alignment < 0.0,
        "counterexample_violates_sufficient_condition": not bool(
            adverse_report["signal_to_noise_condition_holds"]
        ),
    }
    return {
        "artifact": "CauchyLift Phase 2 finite-distribution stochastic suite",
        "run_id": "phase2-stochastic-20260828",
        "theory_constants": {"gamma": 1.0 / math.sqrt(3.0)},
        "benign_unbiased_distribution": {
            "support_size": len(benign),
            "report": benign_report,
            "interior_bias": bias,
            "smooth_descent": descent,
        },
        "expected_ascent_counterexample": {
            "support": [
                {"probability": probability, "gradient": matrix}
                for probability, matrix in adverse
            ],
            "mean_gradient": adverse_mu,
            "expected_direction": adverse_expected_direction,
            "alignment": adverse_alignment,
            "report": adverse_report,
            "interpretation": "unbiasedness alone does not control a normalized nonlinear direction",
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
