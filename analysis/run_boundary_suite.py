#!/usr/bin/env python3
"""Boundary, exhaustive-small-shape, and interior-sensitivity checks.

The bounds checked here are proved in the Phase 2 manuscript. Numerical cases
are regressions and sharpness diagnostics, not substitutes for those proofs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from pathlib import Path

from cauchylift_math import cauchylift, cosine, flatten, frobenius_norm

Matrix = list[list[float]]


def normalize(matrix: Matrix) -> Matrix:
    norm = frobenius_norm(matrix)
    return [[value / norm for value in row] for row in matrix]


def subtract(left: Matrix, right: Matrix) -> Matrix:
    return [[x - y for x, y in zip(row, other)] for row, other in zip(left, right)]


def energy_on_unit_sphere(matrix: Matrix) -> Matrix:
    unit = normalize(matrix)
    rows, columns = len(unit), len(unit[0])
    row = [math.fsum(value * value for value in values) for values in unit]
    column = [
        math.fsum(unit[i][j] * unit[i][j] for i in range(rows)) for j in range(columns)
    ]
    return [[2.0 - row[i] - column[j] for j in range(columns)] for i in range(rows)]


def exhaustive_small_shapes() -> dict[str, object]:
    alphabet = (-1.0, 0.0, 1.0)
    shapes = ((1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2))
    tested = 0
    minimum_cosine = 1.0
    maximum_norm_error = 0.0
    for rows, columns in shapes:
        for values in itertools.product(alphabet, repeat=rows * columns):
            if not any(values):
                continue
            matrix = [list(values[i * columns : (i + 1) * columns]) for i in range(rows)]
            direction = cauchylift(matrix)
            minimum_cosine = min(minimum_cosine, cosine(matrix, direction))
            maximum_norm_error = max(
                maximum_norm_error,
                abs(frobenius_norm(direction) - math.sqrt(min(rows, columns))),
            )
            tested += 1
    return {
        "alphabet": list(alphabet),
        "shapes": [list(shape) for shape in shapes],
        "tested_nonzero_matrices": tested,
        "minimum_cosine": minimum_cosine,
        "maximum_norm_error": maximum_norm_error,
    }


def random_matrix(rng: random.Random, rows: int, columns: int) -> Matrix:
    return [
        [
            0.0
            if rng.random() < 0.2
            else rng.choice((-1.0, 1.0)) * math.exp(rng.uniform(-18.0, 3.0))
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]


def property_cases(seed: int, samples: int) -> dict[str, object]:
    rng = random.Random(seed)
    minimum_cosine = 1.0
    maximum_norm_error = 0.0
    tested = 0
    for _ in range(samples):
        rows, columns = rng.randint(1, 6), rng.randint(1, 6)
        matrix = random_matrix(rng, rows, columns)
        if frobenius_norm(matrix) == 0.0:
            matrix[rng.randrange(rows)][rng.randrange(columns)] = 1.0
        direction = cauchylift(matrix)
        minimum_cosine = min(minimum_cosine, cosine(matrix, direction))
        maximum_norm_error = max(
            maximum_norm_error,
            abs(frobenius_norm(direction) - math.sqrt(min(rows, columns))),
        )
        tested += 1
    return {
        "seed": seed,
        "samples": tested,
        "minimum_cosine": minimum_cosine,
        "maximum_norm_error": maximum_norm_error,
    }


def boundary_continuity_cases() -> dict[str, object]:
    cases: list[dict[str, float]] = []
    for tau in (1e-1, 1e-2, 1e-4, 1e-8, 1e-12):
        # The dominant cell has energy share 1-tau. Off-boundary energy is
        # distributed across a same-row, same-column, and transverse cell.
        off = math.sqrt(tau / 3.0)
        matrix = [[math.sqrt(1.0 - tau), off], [off, off]]
        direction = cauchylift(matrix, radius=1.0)
        boundary = [[1.0, 0.0], [0.0, 0.0]]
        error = frobenius_norm(subtract(direction, boundary))
        bound = 2.0 * (tau / (1.0 - tau)) ** 1.5
        energy = energy_on_unit_sphere(matrix)
        active_denominators = [
            energy[i][j]
            for i in range(2)
            for j in range(2)
            if matrix[i][j] != 0.0
        ]
        cases.append(
            {
                "tau": tau,
                "direction_error": error,
                "proved_upper_bound": bound,
                "error_to_bound_ratio": error / bound if bound else 0.0,
                "active_denominator_ratio": max(active_denominators)
                / min(active_denominators),
                "proved_denominator_ratio_upper_bound": 2.0 / tau,
            }
        )
    return {
        "definition": "tau is total normalized energy outside the dominant cell",
        "cases": cases,
        "all_continuity_bounds_hold": all(
            case["direction_error"] <= case["proved_upper_bound"] * (1.0 + 1e-12)
            for case in cases
        ),
        "all_denominator_bounds_hold": all(
            case["active_denominator_ratio"]
            <= case["proved_denominator_ratio_upper_bound"] * (1.0 + 1e-12)
            for case in cases
        ),
    }


def interior_lipschitz_cases(seed: int, samples: int, delta: float = 0.1) -> dict[str, object]:
    rng = random.Random(seed)
    proved_constant = 4.0 / delta + 16.0 / (delta * delta)
    maximum_observed_ratio = 0.0
    accepted = 0
    attempts = 0
    while accepted < samples and attempts < samples * 100:
        attempts += 1
        x = normalize([[rng.gauss(0.0, 1.0) for _ in range(3)] for _ in range(3)])
        perturbation = [[rng.gauss(0.0, 1e-4) for _ in range(3)] for _ in range(3)]
        y = normalize([[xv + dv for xv, dv in zip(row, delta_row)] for row, delta_row in zip(x, perturbation)])
        if min(flatten(energy_on_unit_sphere(x))) < delta:
            continue
        if min(flatten(energy_on_unit_sphere(y))) < delta:
            continue
        input_distance = frobenius_norm(subtract(x, y))
        output_distance = frobenius_norm(
            subtract(cauchylift(x, radius=1.0), cauchylift(y, radius=1.0))
        )
        maximum_observed_ratio = max(maximum_observed_ratio, output_distance / input_distance)
        accepted += 1
    return {
        "seed": seed,
        "delta": delta,
        "samples": accepted,
        "proved_lipschitz_constant": proved_constant,
        "maximum_observed_ratio": maximum_observed_ratio,
        "all_cases_within_bound": accepted == samples and maximum_observed_ratio <= proved_constant,
    }


def zero_discontinuity() -> dict[str, object]:
    x = [[1.0, 0.0], [0.0, 0.0]]
    y = [[0.0, 1.0], [0.0, 0.0]]
    output_distance = frobenius_norm(
        subtract(cauchylift(x, radius=1.0), cauchylift(y, radius=1.0))
    )
    input_distances = []
    for scale in (1e-1, 1e-4, 1e-8, 1e-16):
        scaled_x = [[scale * value for value in row] for row in x]
        scaled_y = [[scale * value for value in row] for row in y]
        input_distances.append(frobenius_norm(subtract(scaled_x, scaled_y)))
    return {
        "sequence_input_distances": input_distances,
        "constant_output_distance": output_distance,
        "conclusion": "the scale-invariant nonzero map has no continuous extension at zero",
    }


def run(seed: int, samples: int) -> dict[str, object]:
    exhaustive = exhaustive_small_shapes()
    properties = property_cases(seed, samples)
    boundary = boundary_continuity_cases()
    lipschitz = interior_lipschitz_cases(seed + 1, min(samples, 2000))
    zero = zero_discontinuity()
    angle_floor = 1.0 / math.sqrt(3.0)
    checks = {
        "exhaustive_angle": exhaustive["minimum_cosine"] > angle_floor,
        "exhaustive_norm": exhaustive["maximum_norm_error"] <= 1e-14,
        "property_angle": properties["minimum_cosine"] > angle_floor,
        "property_norm": properties["maximum_norm_error"] <= 2e-12,
        "boundary_continuity": boundary["all_continuity_bounds_hold"],
        "denominator_dynamic_range": boundary["all_denominator_bounds_hold"],
        "interior_lipschitz": lipschitz["all_cases_within_bound"],
        "zero_counterexample": zero["constant_output_distance"] > 1.0,
    }
    return {
        "artifact": "CauchyLift Phase 2 boundary and sensitivity suite",
        "run_id": "phase2-boundary-20260828",
        "seed": seed,
        "angle_equality_case": "none; the bound is strict for every nonzero matrix",
        "exhaustive_small_shapes": exhaustive,
        "property_cases": properties,
        "one_sparse_boundary": boundary,
        "interior_lipschitz": lipschitz,
        "outside_proved_region_counterexample": zero,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.seed, arguments.samples)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
