"""Pure Python mathematical implementation of the CauchyLift operator.

Standard library only, zero external dependencies.
Uses nested Python lists so that every arithmetic operation remains inspectable.
"""

from __future__ import annotations

import math
from typing import Sequence

Matrix = list[list[float]]


def as_matrix(values: Sequence[Sequence[float]]) -> Matrix:
    """Validate and convert rectangular matrix values."""
    matrix = [[float(value) for value in row] for row in values]
    if not matrix or not matrix[0]:
        raise ValueError("A nonempty rectangular matrix is required")
    width = len(matrix[0])
    if any(len(row) != width for row in matrix):
        raise ValueError("Ragged matrices are not supported")
    if any(not math.isfinite(value) for row in matrix for value in row):
        raise ValueError("All entries must be finite")
    return matrix


def frobenius_norm(matrix: Sequence[Sequence[float]]) -> float:
    """Compute Frobenius norm of a matrix."""
    return math.sqrt(math.fsum(x * x for row in matrix for x in row))


def inner_product(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> float:
    """Compute Frobenius inner product <A, B> = tr(A^T B)."""
    return math.fsum(x * y for r1, r2 in zip(left, right) for x, y in zip(r1, r2))


def fiber_rms_denominator(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Compute D_ij = RMS(row_i) + RMS(col_j)."""
    mat = as_matrix(matrix)
    rows, cols = len(mat), len(mat[0])

    row_energy = [math.fsum(x * x for x in row) for row in mat]
    col_energy = [math.fsum(mat[i][j] * mat[i][j] for i in range(rows)) for j in range(cols)]

    row_rms = [math.sqrt(re / cols) for re in row_energy]
    col_rms = [math.sqrt(ce / rows) for ce in col_energy]

    return [[row_rms[i] + col_rms[j] for j in range(cols)] for i in range(rows)]


def cauchylift_direction(matrix: Sequence[Sequence[float]]) -> Matrix:
    """Compute normalized CauchyLift direction U = sqrt(max(m, n)) * Z / ||Z||_F."""
    mat = as_matrix(matrix)
    rows, cols = len(mat), len(mat[0])
    radius = math.sqrt(max(rows, cols))

    non_zero_entries = [(i, j) for i in range(rows) for j in range(cols) if mat[i][j] != 0.0]
    if len(non_zero_entries) == 0:
        return [[0.0] * cols for _ in range(rows)]
    if len(non_zero_entries) == 1:
        i, j = non_zero_entries[0]
        out = [[0.0] * cols for _ in range(rows)]
        out[i][j] = math.copysign(radius, mat[i][j])
        return out

    denom = fiber_rms_denominator(mat)
    raw = [[(mat[i][j] / denom[i][j]) if denom[i][j] > 0 and mat[i][j] != 0.0 else 0.0 for j in range(cols)] for i in range(rows)]
    norm = frobenius_norm(raw)
    if norm == 0.0:
        return [[0.0] * cols for _ in range(rows)]

    scale = radius / norm
    return [[scale * val for val in row] for row in raw]
