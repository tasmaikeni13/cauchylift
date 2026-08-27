"""Pure mathematical implementation of the CauchyLift direction.

This module contains no training loop and has no third-party dependencies.  It
uses nested Python lists so that every arithmetic operation remains inspectable.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

Matrix = list[list[float]]


def _as_matrix(values: Sequence[Sequence[float]]) -> Matrix:
    matrix = [[float(value) for value in row] for row in values]
    if not matrix or not matrix[0]:
        raise ValueError("a nonempty rectangular matrix is required")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("ragged matrices are not supported")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("all entries must be finite")
    return matrix


def frobenius_sq(matrix: Sequence[Sequence[float]]) -> float:
    return math.fsum(value * value for row in matrix for value in row)


def frobenius_norm(matrix: Sequence[Sequence[float]]) -> float:
    return math.sqrt(frobenius_sq(matrix))


def inner(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    return math.fsum(
        x * y for left_row, right_row in zip(left, right) for x, y in zip(left_row, right_row)
    )


def scale(matrix: Sequence[Sequence[float]], factor: float) -> Matrix:
    return [[factor * value for value in row] for row in matrix]


def cotransverse_energy(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Return E_ij = (S-r_i) + (S-c_j), computed after safe rescaling.

    The returned field is dimensionless up to a common positive factor.  That
    factor cancels in the projectively normalized CauchyLift direction.
    """

    gradient = _as_matrix(matrix)
    rows, columns = len(gradient), len(gradient[0])
    maximum = max(abs(value) for row in gradient for value in row)
    if maximum == 0.0:
        return [[0.0] * columns for _ in range(rows)]
    normalized = [[value / maximum for value in row] for row in gradient]
    total = frobenius_sq(normalized)
    row_energy = [math.fsum(value * value for value in row) for row in normalized]
    column_energy = [
        math.fsum(normalized[i][j] * normalized[i][j] for i in range(rows))
        for j in range(columns)
    ]
    return [
        [max(0.0, 2.0 * total - row_energy[i] - column_energy[j]) for j in range(columns)]
        for i in range(rows)
    ]


def raw_cauchylift(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Return a numerically scaled representative of G_ij / E_ij.

    Common scaling is irrelevant because :func:`cauchylift` normalizes the
    result.  Multiplying every reciprocal by the smallest positive denominator
    avoids overflow.  If the projective boundary E_ij=0 is reached, the exact
    epsilon-to-zero limit is supported on those active boundary entries.
    """

    gradient = _as_matrix(matrix)
    rows, columns = len(gradient), len(gradient[0])
    maximum = max(abs(value) for row in gradient for value in row)
    if maximum == 0.0:
        return [[0.0] * columns for _ in range(rows)]
    normalized = [[value / maximum for value in row] for row in gradient]
    energy = cotransverse_energy(normalized)

    boundary = [
        (i, j)
        for i in range(rows)
        for j in range(columns)
        if normalized[i][j] != 0.0 and energy[i][j] == 0.0
    ]
    if boundary:
        active = set(boundary)
        return [
            [normalized[i][j] if (i, j) in active else 0.0 for j in range(columns)]
            for i in range(rows)
        ]

    positive = [
        energy[i][j]
        for i in range(rows)
        for j in range(columns)
        if normalized[i][j] != 0.0 and energy[i][j] > 0.0
    ]
    common = min(positive)
    return [
        [
            normalized[i][j] * common / energy[i][j] if normalized[i][j] != 0.0 else 0.0
            for j in range(columns)
        ]
        for i in range(rows)
    ]


def cauchylift(matrix: Sequence[Sequence[float]], radius: float | None = None) -> Matrix:
    """Compute the projectively normalized CauchyLift direction.

    The default radius is sqrt(min(m, n)), matching the Frobenius norm of a
    full-rank rectangular partial isometry.  A zero gradient maps to zero.
    """

    gradient = _as_matrix(matrix)
    rows, columns = len(gradient), len(gradient[0])
    raw = raw_cauchylift(gradient)
    norm = frobenius_norm(raw)
    if norm == 0.0:
        return [[0.0] * columns for _ in range(rows)]
    target = math.sqrt(min(rows, columns)) if radius is None else float(radius)
    if not math.isfinite(target) or target <= 0.0:
        raise ValueError("radius must be finite and positive")
    return scale(raw, target / norm)


def cosine(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    denominator = frobenius_norm(left) * frobenius_norm(right)
    if denominator == 0.0:
        raise ValueError("cosine is undefined for a zero matrix")
    return inner(left, right) / denominator


def transpose(matrix: Sequence[Sequence[float]]) -> Matrix:
    return [list(column) for column in zip(*matrix)]


def flatten(matrix: Sequence[Sequence[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def maximum_absolute_error(left: Iterable[float], right: Iterable[float]) -> float:
    return max((abs(x - y) for x, y in zip(left, right)), default=0.0)
