from __future__ import annotations

import math
from typing import Any

import torch

from .common import SUPPORTED_DTYPES, diagnostics_from, matrixize


def cauchylift_oracle(
    gradient: torch.Tensor, *, return_diagnostics: bool = False
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Slow independent FP64 CPU oracle implementing optimizer_v0.2.json.

    Exclusions are summed directly over all rows/columns other than the target,
    deliberately avoiding the optimized reference's prefix/suffix algorithm.
    The returned tensor is CPU FP64 and has the input shape.
    """
    if gradient.layout != torch.strided:
        gradient = gradient.to_dense()
    if gradient.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported gradient dtype: {gradient.dtype}")
    if gradient.numel() == 0:
        raise ValueError("CauchyLift does not support empty trainable tensors")
    g = matrixize(gradient.detach()).to(device="cpu", dtype=torch.float64).contiguous()
    if not bool(torch.isfinite(g).all()):
        raise ValueError("CauchyLift rejects nonfinite gradients")

    active = g != 0
    count = int(active.sum())
    radius = math.sqrt(max(g.shape))
    direction = torch.zeros_like(g)
    denominators: torch.Tensor | None = None

    if count == 0:
        zero, boundary = 1, 0
    elif count == 1:
        direction[active] = g[active].sign() * radius
        zero, boundary = 0, 1
    else:
        zero, boundary = 0, 0
        m, n = g.shape
        squares = g.square()
        row_energy = squares.sum(dim=1, keepdim=True)
        col_energy = squares.sum(dim=0, keepdim=True)
        row_rms = (row_energy / n).sqrt()
        col_rms = (col_energy / m).sqrt()
        denominators = row_rms + col_rms
        mask = (denominators > 0) & active
        raw = torch.zeros_like(g)
        raw[mask] = g[mask] / denominators[mask]
        raw_norm = torch.linalg.vector_norm(raw)
        if not bool(torch.isfinite(raw_norm)) or float(raw_norm) == 0.0:
            raise FloatingPointError("raw FP64 field cannot be normalized")
        direction = radius * raw / raw_norm


    output = direction.reshape(gradient.shape)
    if not return_diagnostics:
        return output
    diagnostics = diagnostics_from(
        g,
        direction,
        denominators,
        zero=zero,
        boundary=boundary,
        rare_path=0,
        backend="fp64_oracle",
    )
    return output, diagnostics
