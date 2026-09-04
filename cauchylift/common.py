from __future__ import annotations

import math
from typing import Any

import torch


SUPPORTED_DTYPES = {
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
}


def matrixize(tensor: torch.Tensor) -> torch.Tensor:
    """Apply the frozen scalar/vector/first-axis matrixization rule."""
    if tensor.ndim == 0:
        return tensor.reshape(1, 1)
    if tensor.ndim == 1:
        return tensor.reshape(tensor.numel(), 1)
    if tensor.ndim == 2:
        return tensor
    return tensor.reshape(tensor.shape[0], -1)


def radius_for(tensor: torch.Tensor) -> float:
    matrix = matrixize(tensor)
    return math.sqrt(max(matrix.shape))


def diagnostics_from(
    gradient: torch.Tensor,
    direction: torch.Tensor,
    denominators: torch.Tensor | None,
    *,
    zero: int,
    boundary: int,
    rare_path: int,
    backend: str,
) -> dict[str, Any]:
    g = matrixize(gradient.detach()).to(dtype=torch.float64, device="cpu")
    d = matrixize(direction.detach()).to(dtype=torch.float64, device="cpu")
    active = g != 0
    if denominators is not None and bool(active.any()):
        e = denominators.detach().to(dtype=torch.float64, device="cpu")[active]
        positive = e[e > 0]
        e_min = float(positive.min()) if positive.numel() else None
        e_ratio = float(positive.max() / positive.min()) if positive.numel() else None
    else:
        e_min = None
        e_ratio = None
    g_norm = float(torch.linalg.vector_norm(g))
    d_norm = float(torch.linalg.vector_norm(d))
    cosine = float((g * d).sum() / (g_norm * d_norm)) if g_norm and d_norm else None
    energy = d.square()
    total = float(energy.sum())
    row_concentration = float(energy.sum(dim=1).max() / total) if total else 0.0
    col_concentration = float(energy.sum(dim=0).max() / total) if total else 0.0
    return {
        "backend": backend,
        "zero_gradient_count": zero,
        "one_sparse_boundary_count": boundary,
        "fp64_rare_path_count": rare_path,
        "minimum_positive_normalized_denominator": e_min,
        "active_denominator_ratio": e_ratio,
        "gradient_direction_cosine": cosine,
        "update_row_concentration": row_concentration,
        "update_column_concentration": col_concentration,
    }
