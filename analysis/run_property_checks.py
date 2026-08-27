#!/usr/bin/env python3
"""Deterministic algebraic and numerical property checks for CauchyLift."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from cauchylift_math import (
    cauchylift,
    cosine,
    flatten,
    frobenius_norm,
    inner,
    maximum_absolute_error,
    scale,
    transpose,
)


def random_matrix(rng: random.Random, rows: int, columns: int) -> list[list[float]]:
    return [
        [
            rng.choice((-1.0, 1.0)) * math.exp(rng.uniform(-12.0, 12.0))
            if rng.random() > 0.08
            else 0.0
            for _ in range(columns)
        ]
        for _ in range(rows)
    ]


def run(seed: int, samples: int) -> dict[str, object]:
    rng = random.Random(seed)
    shapes = ((1, 1), (1, 7), (2, 2), (2, 9), (5, 3), (8, 8), (17, 5))
    minimum_cosine = 1.0
    maximum_norm_error = 0.0
    maximum_scale_error = 0.0
    maximum_odd_error = 0.0
    maximum_transpose_error = 0.0
    minimum_inner_product = math.inf

    tested = 0
    for _ in range(samples):
        rows, columns = shapes[rng.randrange(len(shapes))]
        gradient = random_matrix(rng, rows, columns)
        if frobenius_norm(gradient) == 0.0:
            gradient[0][0] = 1.0
        direction = cauchylift(gradient)
        target = math.sqrt(min(rows, columns))
        minimum_cosine = min(minimum_cosine, cosine(gradient, direction))
        maximum_norm_error = max(maximum_norm_error, abs(frobenius_norm(direction) - target))
        minimum_inner_product = min(minimum_inner_product, inner(gradient, direction))

        positive_scale = math.exp(rng.uniform(-20.0, 20.0))
        scaled_direction = cauchylift(scale(gradient, positive_scale))
        maximum_scale_error = max(
            maximum_scale_error,
            maximum_absolute_error(flatten(direction), flatten(scaled_direction)),
        )
        odd_direction = cauchylift(scale(gradient, -1.0))
        maximum_odd_error = max(
            maximum_odd_error,
            maximum_absolute_error(flatten(scale(direction, -1.0)), flatten(odd_direction)),
        )
        transpose_direction = cauchylift(transpose(gradient))
        maximum_transpose_error = max(
            maximum_transpose_error,
            maximum_absolute_error(flatten(transpose(direction)), flatten(transpose_direction)),
        )
        tested += 1

    theoretical_floor = 1.0 / math.sqrt(3.0)
    tolerances = {
        "norm": 2e-12,
        "equivariance": 2e-11,
        "angle": 2e-12,
    }
    checks = {
        "fixed_norm": maximum_norm_error <= tolerances["norm"],
        "positive_scale_invariance": maximum_scale_error <= tolerances["equivariance"],
        "oddness": maximum_odd_error <= tolerances["equivariance"],
        "transpose_equivariance": maximum_transpose_error <= tolerances["equivariance"],
        "strict_descent": minimum_inner_product > 0.0,
        "angle_floor": minimum_cosine + tolerances["angle"] >= theoretical_floor,
    }
    return {
        "artifact": "CauchyLift deterministic property checks",
        "seed": seed,
        "samples": tested,
        "theoretical_cosine_floor": theoretical_floor,
        "observed_minimum_cosine": minimum_cosine,
        "observed_minimum_inner_product": minimum_inner_product,
        "maximum_fixed_norm_error": maximum_norm_error,
        "maximum_positive_scale_error": maximum_scale_error,
        "maximum_odd_error": maximum_odd_error,
        "maximum_transpose_error": maximum_transpose_error,
        "checks": checks,
        "all_passed": all(checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--samples", type=int, default=5000)
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
