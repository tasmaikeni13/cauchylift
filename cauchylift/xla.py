"""Google Cloud TPU / XLA native kernel implementation and dispatch for CauchyLift."""

from __future__ import annotations

import math
from typing import Any, Sequence
import torch

from .common import matrixize

_HAS_XLA = False
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    _HAS_XLA = True
except ImportError:
    torch_xla = None
    xm = None


def is_tpu_available() -> bool:
    """Check if Google Cloud TPU / XLA device execution is available."""
    if not _HAS_XLA:
        return False
    try:
        devices = xm.get_xla_supported_devices()
        return len(devices) > 0
    except Exception:
        try:
            dev = torch_xla.device() if hasattr(torch_xla, "device") else xm.xla_device()
            return dev.type == "xla"
        except Exception:
            return False


def get_tpu_device(index: int | None = None) -> torch.device:
    """Return the active TPU XLA device."""
    if not _HAS_XLA:
        raise RuntimeError("torch_xla is not installed")
    if index is not None:
        return xm.xla_device(index)
    if hasattr(torch_xla, "device"):
        return torch_xla.device()
    return xm.xla_device()


def sync_tpu() -> None:
    """Trigger XLA step evaluation barrier on TPU."""
    if _HAS_XLA:
        if hasattr(torch_xla, "sync"):
            torch_xla.sync()
        else:
            xm.mark_step()


def get_hlo_text(tensors: torch.Tensor | Sequence[torch.Tensor]) -> str:
    """Extract lowered XLA HLO module text for the pending tensor computation graphs."""
    if not _HAS_XLA:
        raise RuntimeError("torch_xla is not installed")
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    return torch_xla._XLAC._get_xla_tensors_hlo(list(tensors))


def cauchylift_direction_xla(
    tensor: torch.Tensor,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute the Additive Fiber RMS CauchyLift direction optimized for XLA/TPU HLO compilation.

    Fuses row/col energy reductions, additive fiber RMS scaling, and projective
    Frobenius normalization into a single XLA HLO graph cluster.

    Args:
        tensor: 2D matrix (or matrixized tensor) representing the momentum or velocity.
        accumulation_dtype: Computation dtype for reductions (default float32 on TPU).

    Returns:
        Direction tensor of the same shape and dtype as input, scaled to radius sqrt(max(m, n)).
    """
    orig_shape = tensor.shape
    v_mat = matrixize(tensor).to(accumulation_dtype)
    m, n = v_mat.shape
    radius = math.sqrt(max(m, n))

    squares = v_mat.square()
    row_energy = squares.sum(dim=1, keepdim=True)
    col_energy = squares.sum(dim=0, keepdim=True)

    row_rms = (row_energy / float(n)).sqrt()
    col_rms = (col_energy / float(m)).sqrt()
    denom = row_rms + col_rms

    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    raw = torch.where(denom > 0, v_mat / safe_denom, torch.zeros_like(v_mat))

    norm = torch.linalg.vector_norm(raw)
    safe_norm = torch.where(norm > 0, norm, torch.tensor(1.0, dtype=accumulation_dtype, device=v_mat.device))
    direction = torch.where(norm > 0, (radius / safe_norm) * raw, torch.zeros_like(v_mat))

    return direction.reshape(orig_shape).to(tensor.dtype)


@torch.no_grad()
def cauchylift_xla_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    learning_rate: float,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    nesterov: bool = False,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Execute a single in-place fused CauchyLift optimization step on Google Cloud TPU.

    Lowers and executes:
    1. Historical momentum buffer update: M_t = beta * M_{t-1} + (1 - beta) * G_t
    2. Additive Fiber RMS Cauchy lifting: D_{ij} = RMS(M_{i,:}) + RMS(M_{:,j})
    3. Decoupled weight decay: W = W * (1 - lr * weight_decay)
    4. Curvature-adaptive update: W = W - lr * U
    """
    # 1. Update historical momentum buffer
    momentum_buffer.mul_(momentum).add_(gradient, alpha=1.0 - momentum)

    v = (gradient * (1.0 - momentum) + momentum * momentum_buffer) if nesterov else momentum_buffer

    # 2. Compute CauchyLift direction via XLA-fused fiber RMS
    direction = cauchylift_direction_xla(v, accumulation_dtype=accumulation_dtype)

    # 3. Decoupled weight decay
    if weight_decay != 0.0:
        parameter.mul_(1.0 - learning_rate * weight_decay)

    # 4. In-place parameter update
    parameter.add_(direction.to(parameter.dtype), alpha=-learning_rate)

    return parameter


@torch.no_grad()
def cauchylift_xla_foreach_step_(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    momentum_buffers: list[torch.Tensor],
    learning_rate: float,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    nesterov: bool = False,
    accumulation_dtype: torch.dtype = torch.float32,
    mark_step: bool = False,
) -> None:
    """Execute multi-tensor foreach CauchyLift steps across a parameter group on TPU."""
    if not parameters or len(parameters) != len(gradients) or len(parameters) != len(momentum_buffers):
        raise ValueError("Parameters, gradients, and momentum_buffers must be nonempty equal-length lists")

    for p, g, m in zip(parameters, gradients, momentum_buffers):
        cauchylift_xla_step_(
            p, g, m,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            accumulation_dtype=accumulation_dtype,
        )

    if mark_step and _HAS_XLA:
        xm.mark_step()
