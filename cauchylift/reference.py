"""Pure PyTorch mathematical reference implementation of the CauchyLift optimizer.

CauchyLift is a curvature-adaptive matrix optimizer for deep neural networks.
It combines:
1. Historical momentum filtering to suppress stochastic minibatch gradient noise.
2. Additive Fiber RMS Cauchy lifting:
       D_{ij} = RMS(M_{i,:}) + RMS(M_{:,j})
       Z_{ij} = M_{ij} / D_{ij}
       U = sqrt(max(m, n)) * Z / ||Z||_F
   providing coordinate-wise Riemannian curvature adaptation with O(N^2) complexity.
3. Decoupled weight decay to regulate parameter Frobenius norms and maintain optimal
   conditioning in scale-invariant transformer architectures:
       W_{t+1} = W_t * (1 - lr * weight_decay) - lr * U
"""

from __future__ import annotations

import math
from typing import Any
import torch

from .common import matrixize


def cauchylift_direction(
    tensor: torch.Tensor,
    accumulation_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Compute the projectively normalized CauchyLift direction for a matrix.

    Args:
        tensor: 2D matrix (or matrixized tensor) representing the momentum or velocity.
        accumulation_dtype: Computation dtype for reductions (default float32).

    Returns:
        Direction tensor of the same shape as tensor, normalized to Frobenius radius sqrt(max(m, n)).
    """
    v_mat = matrixize(tensor).to(accumulation_dtype)
    m, n = v_mat.shape
    radius = math.sqrt(max(m, n))

    active = v_mat != 0
    count = int(active.sum().item())

    if count == 0:
        direction = torch.zeros_like(v_mat)
    elif count == 1:
        direction = torch.zeros_like(v_mat)
        direction[active] = v_mat[active].sign() * radius
    else:
        squares = v_mat.square()
        row_energy = squares.sum(dim=1, keepdim=True)
        col_energy = squares.sum(dim=0, keepdim=True)
        row_rms = (row_energy / n).sqrt()
        col_rms = (col_energy / m).sqrt()
        denom = row_rms + col_rms
        mask = (denom > 0) & active
        raw = torch.zeros_like(v_mat)
        raw[mask] = v_mat[mask] / denom[mask]
        norm = torch.linalg.vector_norm(raw)
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) == 0.0:
            direction = torch.zeros_like(v_mat)
        else:
            direction = radius * raw / norm

    return direction.reshape(tensor.shape).to(tensor.dtype)


def cauchylift_reference_step(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor | None,
    lr: float,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    nesterov: bool = False,
    accumulation_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure PyTorch reference step implementation of CauchyLift.

    Args:
        parameter: Trainable parameter tensor (modified in-place).
        gradient: Parameter gradient tensor.
        momentum_buffer: Historical momentum buffer tensor or None (will be initialized).
        lr: Learning rate (step size).
        momentum: Momentum factor beta in [0, 1). Default 0.95.
        weight_decay: Decoupled weight decay factor lambda >= 0. Default 0.01.
        nesterov: Whether to use Nesterov accelerated momentum.
        accumulation_dtype: Accumulation dtype for reductions (default float32).

    Returns:
        tuple of (updated_parameter, updated_momentum_buffer)
    """
    if momentum_buffer is None:
        momentum_buffer = torch.zeros_like(gradient)

    # 1. Update historical momentum buffer: M_t = beta * M_{t-1} + (1 - beta) * G_t
    momentum_buffer.mul_(momentum).add_(gradient, alpha=1.0 - momentum)

    v = (gradient * (1.0 - momentum) + momentum * momentum_buffer) if nesterov else momentum_buffer

    # 2. Compute Additive Fiber RMS CauchyLift direction
    direction = cauchylift_direction(v, accumulation_dtype=accumulation_dtype)

    # 3. Decoupled weight decay: W = W * (1 - lr * weight_decay)
    if weight_decay != 0.0:
        parameter.mul_(1.0 - lr * weight_decay)

    # 4. CauchyLift parameter update: W = W - lr * U
    parameter.add_(direction.to(parameter.dtype), alpha=-lr)

    return parameter, momentum_buffer
