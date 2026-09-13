"""Google Cloud TPU / XLA native kernel implementation and dispatch for CauchyLift."""

from __future__ import annotations

import math
from typing import Any, Sequence
import torch

from .common import matrixize

def _auto_configure_tpu_env() -> None:
    """Auto-configure TPU environment variables if running on TPU v4 to prevent stalling."""
    import os
    if os.path.exists("/dev/accel0"):
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        if "TPU_PROCESS_BOUNDS" not in os.environ and "TPU_PROCESS_ADDRESSES" not in os.environ and "TPU_WORKER_HOSTNAMES" not in os.environ:
            os.environ.setdefault("TPU_SKIP_MDS_QUERY", "1")
            os.environ.setdefault("TPU_ACCELERATOR_TYPE", "v4-8")
            os.environ.setdefault("TPU_PROCESS_BOUNDS", "1,1,1")
            os.environ.setdefault("TPU_CHIPS_PER_HOST_BOUNDS", "2,2,1")
            os.environ.setdefault("TPU_WORKER_HOSTNAMES", "10.130.0.10")
            os.environ.setdefault("TPU_WORKER_ID", "0")


_auto_configure_tpu_env()

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
    Frobenius normalization into a single XLA HLO graph cluster with branchless
    clamping optimized for TPU v4 systolic arrays and vector processing units.

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
    denom = (row_rms + col_rms).clamp_min(1e-12)
    raw = v_mat / denom

    norm = torch.linalg.vector_norm(raw).clamp_min(1e-12)
    direction = (radius / norm) * raw

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


def muon_newton_schulz_xla(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz quintic iteration for matrix polar orthogonalization optimized for TPU v4 XLA.

    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315.
    Approximates the nearest semi-orthogonal matrix (polar factor) to G.
    Executes native BF16 matrix multiplications on TPU v4 systolic MXUs,
    with FP32 accumulation for the initial Frobenius normalization to prevent overflow.
    Transposes tall matrices (m > n) to ensure the Gram matrix is of minimal dimension min(m, n) x min(m, n).
    """
    assert G.ndim == 2, f"Expected 2D matrix, got {G.ndim}D"
    orig_shape = G.shape
    orig_dtype = G.dtype
    m, n = orig_shape

    # 1. Numerically stable Frobenius normalization in FP32
    norm = torch.linalg.vector_norm(G.to(torch.float32)).clamp_min(eps)
    X = (G.to(torch.float32) / norm).to(torch.bfloat16 if G.dtype == torch.bfloat16 else torch.float32)

    # 2. Minimal dimension transposition: if m > n, transpose so m <= n
    transposed = False
    if m > n:
        X = X.T
        m, n = n, m
        transposed = True

    # 3. Quintic Newton-Schulz iterations on TPU TensorCores
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        # A = X @ X.T has shape [m, m] where m <= n
        A = torch.matmul(X, X.T)
        A2 = torch.matmul(A, A)
        B = b * A + c * A2
        X = a * X + torch.matmul(B, X)

    # 4. Transpose back if needed
    if transposed:
        X = X.T

    # 5. Aspect ratio scaling: sqrt(max(1.0, orig_m / orig_n))
    scale = math.sqrt(max(1.0, orig_shape[0] / orig_shape[1]))
    X = X * scale

    return X.to(dtype=orig_dtype)


@torch.no_grad()
def muon_xla_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    learning_rate: float,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    nesterov: bool = True,
    ns_steps: int = 5,
) -> torch.Tensor:
    """Execute a single in-place fused Muon optimization step on TPU v4.

    1. Updates momentum buffer: M_t = beta * M_{t-1} + G_t
    2. Computes velocity: V_t = G_t + beta * M_t if Nesterov else M_t
    3. Computes orthogonal polar factor via TPU-optimized Newton-Schulz 5
    4. Decoupled weight decay: W = W * (1 - lr * weight_decay)
    5. In-place parameter update: W = W - lr * U
    """
    # 1. Update historical momentum buffer
    momentum_buffer.mul_(momentum).add_(gradient)

    v = gradient + momentum * momentum_buffer if nesterov else momentum_buffer

    # 2. Compute Newton-Schulz orthogonalized direction on TPU
    direction = muon_newton_schulz_xla(v, steps=ns_steps)

    # 3. Decoupled weight decay
    if weight_decay != 0.0:
        parameter.mul_(1.0 - learning_rate * weight_decay)

    # 4. Parameter update
    parameter.add_(direction.to(parameter.dtype), alpha=-learning_rate)
    return parameter


@torch.no_grad()
def muon_xla_foreach_step_(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    momentum_buffers: list[torch.Tensor],
    learning_rate: float,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
    nesterov: bool = True,
    ns_steps: int = 5,
    mark_step: bool = False,
) -> None:
    """Execute multi-tensor foreach Muon steps across a parameter group on TPU."""
    if not parameters or len(parameters) != len(gradients) or len(parameters) != len(momentum_buffers):
        raise ValueError("Parameters, gradients, and momentum_buffers must be nonempty equal-length lists")

    for p, g, m in zip(parameters, gradients, momentum_buffers):
        muon_xla_step_(
            p, g, m,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            ns_steps=ns_steps,
        )

    if mark_step and _HAS_XLA:
        xm.mark_step()


@torch.no_grad()
def adamw_xla_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    step: int | torch.Tensor,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
) -> torch.Tensor:
    """Execute single-parameter AdamW update optimized for XLA on TPU (for 1D / embedding parameters)."""
    g_fp32 = gradient.to(torch.float32)

    exp_avg.mul_(beta1).add_(g_fp32, alpha=1.0 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1.0 - beta2)

    step_val = int(step.item()) if isinstance(step, torch.Tensor) and step.numel() == 1 and step.device.type == "cpu" else step
    if isinstance(step_val, (int, float)):
        bias_correction1 = 1.0 - beta1 ** step_val
        bias_correction2 = 1.0 - beta2 ** step_val
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        step_size = learning_rate / bias_correction1
    else:
        bias_correction1 = 1.0 - beta1 ** step_val
        bias_correction2 = 1.0 - beta2 ** step_val
        denom = (exp_avg_sq.sqrt() / bias_correction2.sqrt()).add_(eps)
        step_size = learning_rate / bias_correction1

    if weight_decay != 0.0:
        parameter.mul_(1.0 - learning_rate * weight_decay)

    update = exp_avg / denom
    parameter.add_(update.to(parameter.dtype), alpha=-step_size)
    return parameter


@torch.no_grad()
def adamw_xla_foreach_step_(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    exp_avgs: list[torch.Tensor],
    exp_avg_sqs: list[torch.Tensor],
    step: int | torch.Tensor,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.95,
    eps: float = 1e-8,
    weight_decay: float = 0.01,
    mark_step: bool = False,
) -> None:
    """Execute multi-tensor foreach AdamW steps across a parameter group on TPU."""
    if not parameters or len(parameters) != len(gradients) or len(parameters) != len(exp_avgs) or len(parameters) != len(exp_avg_sqs):
        raise ValueError("Parameters, gradients, exp_avgs, and exp_avg_sqs must be nonempty equal-length lists")

    for p, g, ea, eas in zip(parameters, gradients, exp_avgs, exp_avg_sqs):
        adamw_xla_step_(
            p, g, ea, eas,
            step=step,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
        )

    if mark_step and _HAS_XLA:
        xm.mark_step()


