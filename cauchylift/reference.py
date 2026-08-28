from __future__ import annotations

import math
from typing import Any

import torch

from .common import SUPPORTED_DTYPES, diagnostics_from, matrixize


def _exclusion_sum(values: torch.Tensor) -> torch.Tensor:
    """Sum all entries except each target without total-minus-target."""
    zero = torch.zeros(1, dtype=values.dtype, device=values.device)
    prefix = torch.cat((zero, torch.cumsum(values[:-1], dim=0)))
    suffix = torch.cat(
        (torch.flip(torch.cumsum(torch.flip(values[1:], (0,)), dim=0), (0,)), zero)
    )
    return prefix + suffix


def _reference_impl(
    gradient: torch.Tensor,
    accumulation_dtype: torch.dtype,
    *,
    return_diagnostics: bool,
    rare_path: int,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    matrix = matrixize(gradient)
    g = matrix.to(dtype=accumulation_dtype)
    if not bool(torch.isfinite(g).all().item()):
        raise ValueError("CauchyLift rejects nonfinite gradients")
    active = g != 0
    count = int(active.sum().item())
    radius = math.sqrt(min(g.shape))
    direction = torch.zeros_like(g)
    denominators: torch.Tensor | None = None
    zero = int(count == 0)
    boundary = int(count == 1)

    if count == 1:
        direction[active] = g[active].sign() * radius
    elif count > 1:
        scaled = g / g.abs().max()
        squares = scaled.square()
        row_energy = squares.sum(dim=1)
        column_energy = squares.sum(dim=0)
        outside_rows = _exclusion_sum(row_energy)
        outside_columns = _exclusion_sum(column_energy)
        denominators = outside_rows[:, None] + outside_columns[None, :]
        active_denominators = denominators[active]
        if bool((active_denominators <= 0).any().item()):
            if accumulation_dtype != torch.float64:
                return _reference_impl(
                    gradient,
                    torch.float64,
                    return_diagnostics=return_diagnostics,
                    rare_path=rare_path + 1,
                )
            raise FloatingPointError(
                "positive active denominator was not representable in FP64"
            )
        e_min = active_denominators.min()
        raw = torch.zeros_like(g)
        raw[active] = scaled[active] * e_min / active_denominators
        norm = torch.linalg.vector_norm(raw)
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) == 0.0:
            if accumulation_dtype != torch.float64:
                return _reference_impl(
                    gradient,
                    torch.float64,
                    return_diagnostics=return_diagnostics,
                    rare_path=rare_path + 1,
                )
            raise FloatingPointError("raw FP64 field cannot be normalized")
        direction = radius * raw / norm

    output = direction.reshape(gradient.shape)
    if not return_diagnostics:
        return output
    diagnostics = diagnostics_from(
        g,
        direction,
        denominators,
        zero=zero,
        boundary=boundary,
        rare_path=rare_path,
        backend=f"pytorch_{str(accumulation_dtype).removeprefix('torch.')}",
    )
    return output, diagnostics


def cauchylift_reference(
    gradient: torch.Tensor,
    *,
    return_diagnostics: bool = False,
    accumulation_dtype: torch.dtype | None = None,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Clear exclusion-safe PyTorch implementation of the frozen map."""
    if gradient.layout != torch.strided:
        gradient = gradient.to_dense()
    if gradient.dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported gradient dtype: {gradient.dtype}")
    if gradient.numel() == 0:
        raise ValueError("CauchyLift does not support empty trainable tensors")
    if accumulation_dtype is None:
        accumulation_dtype = (
            torch.float64 if gradient.dtype == torch.float64 else torch.float32
        )
    if accumulation_dtype not in (torch.float32, torch.float64):
        raise ValueError("accumulation_dtype must be float32 or float64")
    return _reference_impl(
        gradient,
        accumulation_dtype,
        return_diagnostics=return_diagnostics,
        rare_path=0,
    )
