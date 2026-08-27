#!/usr/bin/env python3
"""Numerical checks behind two documented candidate rejections."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path


def halley_ratio(kappa: float) -> float:
    return kappa * (kappa * kappa + 3.0) / (3.0 * kappa * kappa + 1.0)


def triple_coth_ratio(kappa: float) -> float:
    # If kappa = coth(x), then coth(3x) has this rational form.
    x = 0.5 * math.log((kappa + 1.0) / (kappa - 1.0))
    return math.cosh(3.0 * x) / math.sinh(3.0 * x)


def sign(value: float) -> float:
    return 1.0 if value >= 0.0 else -1.0


def plaquette_dual(gradient: list[float]) -> list[float]:
    a, b, c, d = gradient
    diagonal_product = abs(a * d)
    off_product = abs(b * c)
    total = diagonal_product + off_product
    if total == 0.0:
        return [sign(value) for value in gradient]
    diagonal_weight = math.sqrt(2.0 * off_product / total)
    off_weight = math.sqrt(2.0 * diagonal_product / total)
    return [
        sign(a) * diagonal_weight,
        sign(b) * off_weight,
        sign(c) * off_weight,
        sign(d) * diagonal_weight,
    ]


def exact_line_iterations(
    hessian: list[float], initial: list[float], direction_name: str, limit: int = 1000
) -> int:
    point = initial[:]
    initial_value = 0.5 * math.fsum(h * x * x for h, x in zip(hessian, point))
    for iteration in range(1, limit + 1):
        gradient = [h * x for h, x in zip(hessian, point)]
        if direction_name == "sign":
            direction = [sign(value) for value in gradient]
        else:
            direction = plaquette_dual(gradient)
        numerator = math.fsum(g * d for g, d in zip(gradient, direction))
        denominator = math.fsum(h * d * d for h, d in zip(hessian, direction))
        step = numerator / denominator
        point = [x - step * d for x, d in zip(point, direction)]
        value = 0.5 * math.fsum(h * x * x for h, x in zip(hessian, point))
        if value <= initial_value * 1e-8:
            return iteration
    return limit


def run(seed: int, trials: int) -> dict[str, object]:
    rng = random.Random(seed)
    maximum_identity_error = max(
        abs(halley_ratio(kappa) - triple_coth_ratio(kappa))
        for kappa in (1.0001, 1.01, 1.1, 2.0, 10.0, 1e3)
    )
    results = {"sign": [], "plaquette_dual": []}
    for _ in range(trials):
        hessian = [10.0 ** rng.uniform(0.0, 4.0) for _ in range(4)]
        initial = [rng.gauss(0.0, 1.0) for _ in range(4)]
        for name in results:
            results[name].append(exact_line_iterations(hessian, initial, name))
    return {
        "artifact": "Rejected-candidate checks",
        "seed": seed,
        "cofactor_exterior_branch": {
            "maximum_halley_triple_angle_identity_error": maximum_identity_error,
            "decision": (
                "rejected: the proposed condition-number map is the Halley/polar triple-angle map, "
                "not a distinct optimizer primitive"
            ),
        },
        "plaquette_cross_ratio_dual_branch": {
            name: {
                "median_iterations": statistics.median(values),
                "failures_at_1000": sum(value == 1000 for value in values),
            }
            for name, values in results.items()
        },
        "plaquette_decision": (
            "rejected: exact cross-ratio inversion frequently destroys useful alignment; making it safe "
            "requires a conventional blend that violates the research contract"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.seed, arguments.trials)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
