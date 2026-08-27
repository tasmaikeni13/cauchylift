#!/usr/bin/env python3
"""Exact algebraic-rank and floating stable-rank probes for the Cauchy law."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from fractions import Fraction
from pathlib import Path


def exact_rank(matrix: list[list[Fraction]]) -> int:
    work = [row[:] for row in matrix]
    rows, columns = len(work), len(work[0])
    rank = 0
    for column in range(columns):
        pivot = next((i for i in range(rank, rows) if work[i][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        divisor = work[rank][column]
        work[rank] = [value / divisor for value in work[rank]]
        for i in range(rows):
            if i == rank:
                continue
            factor = work[i][column]
            work[i] = [x - factor * y for x, y in zip(work[i], work[rank])]
        rank += 1
        if rank == rows:
            break
    return rank


def exact_raw_lift(left: list[int], right: list[int]) -> list[list[Fraction]]:
    left_energy = sum(value * value for value in left)
    right_energy = sum(value * value for value in right)
    total = left_energy * right_energy
    return [
        [
            Fraction(
                left[i] * right[j],
                2 * total
                - left[i] * left[i] * right_energy
                - right[j] * right[j] * left_energy,
            )
            for j in range(len(right))
        ]
        for i in range(len(left))
    ]


def floating_raw_lift(left: list[float], right: list[float]) -> list[list[float]]:
    left_energy = math.fsum(value * value for value in left)
    right_energy = math.fsum(value * value for value in right)
    total = left_energy * right_energy
    return [
        [
            left[i]
            * right[j]
            / (
                2.0 * total
                - left[i] * left[i] * right_energy
                - right[j] * right[j] * left_energy
            )
            for j in range(len(right))
        ]
        for i in range(len(left))
    ]


def stable_rank(matrix: list[list[float]]) -> float:
    rows, columns = len(matrix), len(matrix[0])
    vector = [1.0 / math.sqrt(columns)] * columns
    image: list[float] = []
    for _ in range(120):
        image = [math.fsum(matrix[i][j] * vector[j] for j in range(columns)) for i in range(rows)]
        back = [math.fsum(matrix[i][j] * image[i] for i in range(rows)) for j in range(columns)]
        length = math.sqrt(math.fsum(value * value for value in back))
        vector = [value / length for value in back]
    spectral_sq = math.fsum(value * value for value in image)
    frobenius_sq = math.fsum(value * value for row in matrix for value in row)
    return frobenius_sq / spectral_sq


def run(seed: int, samples: int) -> dict[str, object]:
    left = [1, 2, 3, 5]
    right = [1, 3, 4, 7]
    exact = exact_raw_lift(left, right)
    rng = random.Random(seed)
    floating: dict[str, object] = {}
    for size in (4, 8, 16, 32):
        values = []
        for _ in range(samples):
            u = [rng.choice((-1.0, 1.0)) * math.exp(rng.gauss(0.0, 1.0)) for _ in range(size)]
            v = [rng.choice((-1.0, 1.0)) * math.exp(rng.gauss(0.0, 1.0)) for _ in range(size)]
            values.append(stable_rank(floating_raw_lift(u, v)))
        floating[str(size)] = {
            "samples": samples,
            "median_output_stable_rank": statistics.median(values),
            "mean_output_stable_rank": statistics.fmean(values),
            "maximum_output_stable_rank": max(values),
            "input_stable_rank": 1.0,
        }

    x1, x2 = Fraction(2, 3), Fraction(3, 5)
    y1, y2 = Fraction(4, 7), Fraction(5, 8)
    determinant_direct = Fraction(1, x1 + y1) * Fraction(1, x2 + y2) - Fraction(
        1, x1 + y2
    ) * Fraction(1, x2 + y1)
    determinant_formula = (x1 - x2) * (y1 - y2) / (
        (x1 + y1) * (x1 + y2) * (x2 + y1) * (x2 + y2)
    )
    return {
        "artifact": "CauchyLift rank probe",
        "seed": seed,
        "exact_example": {
            "shape": [len(left), len(right)],
            "input_outer_product_rank": 1,
            "output_exact_rank": exact_rank(exact),
            "left_factors": left,
            "right_factors": right,
        },
        "two_by_two_cauchy_determinant_identity": {
            "direct": str(determinant_direct),
            "factorized": str(determinant_formula),
            "equal": determinant_direct == determinant_formula,
            "nonzero": determinant_direct != 0,
        },
        "floating_stable_rank": floating,
        "interpretation": (
            "Generic exact rank can be full while floating stable rank remains close to one. "
            "The algebraic rank theorem must not be marketed as evidence of useful spectral diversity."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.seed, arguments.samples)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
