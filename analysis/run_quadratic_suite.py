#!/usr/bin/env python3
"""Small deterministic quadratic probes; this is not a training benchmark.

The suite separates direction quality (exact line search) from practical step
sensitivity (a held-out, oracle-selected inverse-square-root schedule).  The
second probe is intentionally included because it exposes a current weakness of
CauchyLift instead of letting an exact line search conceal it.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from cauchylift_math import cauchylift, frobenius_norm, inner, scale

Matrix = list[list[float]]
Direction = Callable[[Matrix], Matrix]


def transpose(matrix: Matrix) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def matmul(left: Matrix, right: Matrix) -> Matrix:
    right_t = transpose(right)
    return [
        [math.fsum(x * y for x, y in zip(row, column)) for column in right_t]
        for row in left
    ]


def diagonal(values: list[float]) -> Matrix:
    return [[value if i == j else 0.0 for j, value in enumerate(values)] for i in range(len(values))]


def orthogonal_matrix(rng: random.Random, size: int) -> Matrix:
    columns: list[list[float]] = []
    for j in range(size):
        vector = [rng.gauss(0.0, 1.0) for _ in range(size)]
        for column in columns:
            projection = math.fsum(x * y for x, y in zip(vector, column))
            vector = [x - projection * y for x, y in zip(vector, column)]
        length = math.sqrt(math.fsum(value * value for value in vector))
        if length < 1e-14:
            vector = [1.0 if i == j else 0.0 for i in range(size)]
            length = 1.0
        columns.append([value / length for value in vector])
    return transpose(columns)


def symmetric_eigendecomposition(matrix: Matrix) -> tuple[list[float], Matrix]:
    """Cyclic Jacobi eigendecomposition for the tiny analysis matrices."""

    size = len(matrix)
    work = [row[:] for row in matrix]
    vectors = [[1.0 if i == j else 0.0 for j in range(size)] for i in range(size)]
    for _ in range(80 * size * size):
        p, q = max(
            ((i, j) for i in range(size) for j in range(i + 1, size)),
            key=lambda pair: abs(work[pair[0]][pair[1]]),
        )
        if abs(work[p][q]) <= 1e-13 * max(1.0, max(abs(work[i][i]) for i in range(size))):
            break
        angle = 0.5 * math.atan2(2.0 * work[p][q], work[q][q] - work[p][p])
        cosine, sine = math.cos(angle), math.sin(angle)
        for k in range(size):
            old_p, old_q = work[p][k], work[q][k]
            work[p][k] = cosine * old_p - sine * old_q
            work[q][k] = sine * old_p + cosine * old_q
        for k in range(size):
            old_p, old_q = work[k][p], work[k][q]
            work[k][p] = cosine * old_p - sine * old_q
            work[k][q] = sine * old_p + cosine * old_q
        for k in range(size):
            old_p, old_q = vectors[k][p], vectors[k][q]
            vectors[k][p] = cosine * old_p - sine * old_q
            vectors[k][q] = sine * old_p + cosine * old_q
    return [max(0.0, work[i][i]) for i in range(size)], vectors


def normalized_gradient(gradient: Matrix) -> Matrix:
    radius = math.sqrt(min(len(gradient), len(gradient[0])))
    return scale(gradient, radius / frobenius_norm(gradient))


def sign_direction(gradient: Matrix) -> Matrix:
    signed = [[1.0 if value >= 0.0 else -1.0 for value in row] for row in gradient]
    return normalized_gradient(signed)


def sinkhorn_direction(gradient: Matrix, rounds: int = 5) -> Matrix:
    work = [row[:] for row in gradient]
    rows, columns = len(work), len(work[0])
    for _ in range(rounds):
        row_norms = [math.sqrt(math.fsum(value * value for value in row)) for row in work]
        work = [
            [value / row_norms[i] if row_norms[i] else 0.0 for value in row]
            for i, row in enumerate(work)
        ]
        column_norms = [
            math.sqrt(math.fsum(work[i][j] * work[i][j] for i in range(rows)))
            for j in range(columns)
        ]
        work = [
            [work[i][j] / column_norms[j] if column_norms[j] else 0.0 for j in range(columns)]
            for i in range(rows)
        ]
    return normalized_gradient(work)


def polar_direction(gradient: Matrix) -> Matrix:
    gram = matmul(transpose(gradient), gradient)
    eigenvalues, eigenvectors = symmetric_eigendecomposition(gram)
    largest = max(eigenvalues, default=0.0)
    cutoff = largest * 1e-12
    inverse_root = diagonal(
        [1.0 / math.sqrt(value) if value > cutoff else 0.0 for value in eigenvalues]
    )
    polar = matmul(gradient, matmul(eigenvectors, matmul(inverse_root, transpose(eigenvectors))))
    if frobenius_norm(polar) == 0.0:
        return normalized_gradient(gradient)
    return normalized_gradient(polar)


DIRECTIONS: dict[str, Direction] = {
    "normalized_gradient": normalized_gradient,
    "sign": sign_direction,
    "sinkhorn_5": sinkhorn_direction,
    "exact_polar": polar_direction,
    "cauchylift": cauchylift,
}


@dataclass(frozen=True)
class Quadratic:
    left: Matrix
    right: Matrix
    initial: Matrix

    def gradient(self, point: Matrix) -> Matrix:
        return matmul(matmul(self.left, point), self.right)

    def value(self, point: Matrix) -> float:
        return 0.5 * inner(point, self.gradient(point))


def make_problem(
    seed: int, rotated: bool, factor_condition: float, size: int = 4
) -> Quadratic:
    rng = random.Random(seed)
    spectrum = [factor_condition ** (index / (size - 1)) for index in range(size)]
    if rotated:
        left_basis = orthogonal_matrix(rng, size)
        right_basis = orthogonal_matrix(rng, size)
        left = matmul(left_basis, matmul(diagonal(spectrum), transpose(left_basis)))
        right = matmul(right_basis, matmul(diagonal(spectrum), transpose(right_basis)))
    else:
        left = diagonal(spectrum)
        right = diagonal(spectrum)
    initial = [[rng.gauss(0.0, 1.0) for _ in range(size)] for _ in range(size)]
    return Quadratic(left, right, initial)


def subtract_step(point: Matrix, direction: Matrix, step: float) -> Matrix:
    return [
        [value - step * delta for value, delta in zip(row, direction_row)]
        for row, direction_row in zip(point, direction)
    ]


def exact_line_run(
    problem: Quadratic, direction_map: Direction, steps: int, tolerance: float
) -> tuple[int, float]:
    point = [row[:] for row in problem.initial]
    initial_value = problem.value(point)
    ratio = 1.0
    for iteration in range(1, steps + 1):
        gradient = problem.gradient(point)
        direction = direction_map(gradient)
        curvature = inner(direction, problem.gradient(direction))
        step = inner(gradient, direction) / curvature
        point = subtract_step(point, direction, step)
        ratio = problem.value(point) / initial_value
        if ratio <= tolerance:
            return iteration, ratio
    return steps, ratio


def scheduled_run(problem: Quadratic, direction_map: Direction, steps: int, eta: float) -> float:
    point = [row[:] for row in problem.initial]
    initial_value = problem.value(point)
    for iteration in range(1, steps + 1):
        direction = direction_map(problem.gradient(point))
        point = subtract_step(point, direction, eta / math.sqrt(iteration))
    return problem.value(point) / initial_value


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def run(seed: int, trials: int, exact_steps: int, scheduled_steps: int) -> dict[str, object]:
    if trials < 4 or trials % 2:
        raise ValueError("trials must be an even integer of at least four")
    eta_grid = [
        0.001,
        0.00215443469,
        0.00464158883,
        0.01,
        0.0215443469,
        0.0464158883,
        0.1,
        0.215443469,
        0.464158883,
    ]
    report: dict[str, object] = {
        "artifact": "CauchyLift analytic quadratic direction probe",
        "seed": seed,
        "matrix_size": 4,
        "trials": trials,
        "split": "first half schedule tuning; second half held-out reporting",
        "exact_line_search_target_relative_objective": 1e-8,
        "exact_line_search_max_steps": exact_steps,
        "scheduled_steps": scheduled_steps,
        "schedule": "eta / sqrt(t), eta selected on tuning split",
        "conditions": {},
        "interpretation_guardrail": (
            "These are tiny deterministic quadratics, not neural training or wall-clock evidence. "
            "Exact line search is intentionally paired with a held-out scheduled-step probe."
        ),
    }
    split = trials // 2
    for condition_index, factor_condition in enumerate((10.0, 100.0)):
        condition_report: dict[str, object] = {
            "kronecker_hessian_condition_number": factor_condition * factor_condition,
            "geometries": {},
        }
        for geometry_index, rotated in enumerate((False, True)):
            problems = [
                make_problem(
                    seed + condition_index * 1_000_000 + geometry_index * 100_000 + trial,
                    rotated,
                    factor_condition,
                )
                for trial in range(trials)
            ]
            tuning, evaluation = problems[:split], problems[split:]
            method_report: dict[str, object] = {}
            for name, direction_map in DIRECTIONS.items():
                tuning_scores = {
                    eta: statistics.median(
                        math.log10(
                            max(scheduled_run(problem, direction_map, scheduled_steps, eta), 1e-300)
                        )
                        for problem in tuning
                    )
                    for eta in eta_grid
                }
                selected_eta = min(eta_grid, key=tuning_scores.get)
                held_out_log_ratios = [
                    math.log10(
                        max(
                            scheduled_run(problem, direction_map, scheduled_steps, selected_eta),
                            1e-300,
                        )
                    )
                    for problem in evaluation
                ]
                exact = [
                    exact_line_run(problem, direction_map, exact_steps, 1e-8)
                    for problem in evaluation
                ]
                method_report[name] = {
                    "selected_eta": selected_eta,
                    "tuning_median_log10_relative_objective": tuning_scores[selected_eta],
                    "held_out_scheduled_log10_relative_objective": summarize(held_out_log_ratios),
                    "exact_line_search_iterations": summarize([float(item[0]) for item in exact]),
                    "exact_line_search_failures": sum(
                        item[0] == exact_steps and item[1] > 1e-8 for item in exact
                    ),
                    "exact_line_search_final_log10_relative_objective": summarize(
                        [math.log10(max(item[1], 1e-300)) for item in exact]
                    ),
                }
            condition_report["geometries"]["rotated" if rotated else "axis_aligned"] = (
                method_report
            )
        report["conditions"][f"kappa_{int(factor_condition * factor_condition)}"] = condition_report
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--exact-steps", type=int, default=600)
    parser.add_argument("--scheduled-steps", type=int, default=400)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = run(arguments.seed, arguments.trials, arguments.exact_steps, arguments.scheduled_steps)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
