from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

from cauchylift.xla import (
    is_tpu_available,
    muon_newton_schulz_xla,
    muon_xla_step_,
    adamw_xla_step_,
    sync_tpu,
)


def zeropower_via_newtonschulz5(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Newton-Schulz quintic iteration for matrix orthogonalization (Muon).

    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315.
    Approximates the nearest orthogonal matrix (polar factor) to G.
    Uses FP32 accumulation for initial Frobenius normalization and transposes
    tall matrices to minimize Gram matrix dimensionality min(m, n) x min(m, n).
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"
    orig_shape = G.shape
    orig_dtype = G.dtype
    m, n = orig_shape

    # 1. Numerically stable Frobenius normalization
    norm = torch.linalg.vector_norm(G.to(torch.float32)).clamp_min(eps)
    X = (G.to(torch.float32) / norm).to(torch.bfloat16 if G.dtype == torch.bfloat16 else torch.float32)

    # 2. Transposition for minimal dimension squared
    transposed = False
    if m > n:
        X = X.T
        m, n = n, m
        transposed = True

    # 3. Quintic Newton-Schulz iterations
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(steps):
        # A = X @ X.T has shape [m, m]
        A = torch.matmul(X, X.T)
        A2 = torch.matmul(A, A)
        B = b * A + c * A2
        X = a * X + torch.matmul(B, X)

    if transposed:
        X = X.T

    # 4. Aspect ratio scaling: sqrt(max(1.0, orig_m / orig_n))
    scale = math.sqrt(max(1.0, orig_shape[0] / orig_shape[1]))
    X = X * scale

    return X.to(dtype=orig_dtype)


class Muon(Optimizer):
    """Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

    Applies Newton-Schulz quintic orthogonalization to 2D internal weight matrices,
    and standard AdamW updates to 1D and embedding parameters.
    Optimized for Google Cloud TPU v4 (Torch-XLA / PJRT) systolic MXU execution,
    supporting distributed multi-core data-parallel training across 16 TPU chips with 0 drift.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.01,
        adamw_lr: float = 6e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
        backend: str = "auto",
    ) -> None:
        self.backend = backend.lower()

        # Handle input params: automatic partitioning into Muon 2D vs AdamW 1D/embedding
        param_list = list(params)
        if len(param_list) > 0 and isinstance(param_list[0], dict):
            # User passed param_groups directly
            param_groups = param_list
        else:
            # Automatic separation into internal 2D matrices vs 1D / embeddings
            muon_params = []
            adamw_params = []
            for p in param_list:
                if not isinstance(p, torch.nn.Parameter):
                    p = torch.nn.Parameter(p)
                if not p.requires_grad:
                    continue
                # 2D internal matrices (Linear, Attention, MLP)
                if p.ndim == 2 and min(p.shape) > 1 and max(p.shape) < 10000:
                    muon_params.append(p)
                else:
                    adamw_params.append(p)

            param_groups = [
                {
                    "params": muon_params,
                    "lr": lr,
                    "momentum": momentum,
                    "nesterov": nesterov,
                    "ns_steps": ns_steps,
                    "weight_decay": weight_decay,
                    "is_muon": True,
                },
                {
                    "params": adamw_params,
                    "lr": adamw_lr,
                    "betas": adamw_betas,
                    "eps": adamw_eps,
                    "weight_decay": adamw_weight_decay,
                    "is_muon": False,
                },
            ]

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_weight_decay=adamw_weight_decay,
            is_muon=True,
        )
        super().__init__(param_groups, defaults)

    def _is_xla_active(self) -> bool:
        if self.backend in ("xla", "tpu"):
            return True
        if self.backend == "reference":
            return False
        # auto backend: check if any param is on XLA
        for group in self.param_groups:
            for p in group["params"]:
                if p.device.type == "xla":
                    return True
        return False

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        use_xla = self._is_xla_active()

        for group in self.param_groups:
            group_is_muon = group.get("is_muon", None)

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Determine whether this individual parameter uses Muon or AdamW
                if group_is_muon is not None:
                    is_muon_param = group_is_muon
                else:
                    is_muon_param = p.ndim == 2 and min(p.shape) > 1 and max(p.shape) < 10000

                if is_muon_param:
                    # ================= Muon 2D Update =================
                    lr = group["lr"]
                    momentum = group.get("momentum", 0.95)
                    nesterov = group.get("nesterov", True)
                    ns_steps = group.get("ns_steps", 5)
                    weight_decay = group.get("weight_decay", 0.0)

                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    buf = state["momentum_buffer"]

                    if use_xla and p.device.type == "xla":
                        muon_xla_step_(
                            p,
                            grad,
                            buf,
                            learning_rate=lr,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            nesterov=nesterov,
                            ns_steps=ns_steps,
                        )
                    else:
                        buf.mul_(momentum).add_(grad)
                        v = grad + momentum * buf if nesterov else buf
                        update = zeropower_via_newtonschulz5(v, steps=ns_steps)
                        if weight_decay != 0.0:
                            p.mul_(1.0 - lr * weight_decay)
                        p.add_(update, alpha=-lr)

                else:
                    # ================= AdamW 1D/Embedding Update =================
                    lr = group.get("adamw_lr", group.get("lr", 6e-4))
                    beta1, beta2 = group.get("adamw_betas", group.get("betas", (0.9, 0.95)))
                    eps = group.get("adamw_eps", group.get("eps", 1e-8))
                    weight_decay = group.get("adamw_weight_decay", group.get("weight_decay", 0.01))

                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                    state["step"] += 1
                    step_count = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    if use_xla and p.device.type == "xla":
                        adamw_xla_step_(
                            p,
                            grad,
                            exp_avg,
                            exp_avg_sq,
                            step=step_count,
                            learning_rate=lr,
                            beta1=beta1,
                            beta2=beta2,
                            eps=eps,
                            weight_decay=weight_decay,
                        )
                    else:
                        g_fp32 = grad.to(torch.float32)
                        exp_avg.mul_(beta1).add_(g_fp32, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1.0 - beta2)

                        bias_correction1 = 1.0 - beta1 ** step_count
                        bias_correction2 = 1.0 - beta2 ** step_count

                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                        step_size = lr / bias_correction1

                        if weight_decay != 0.0:
                            p.mul_(1.0 - lr * weight_decay)

                        p.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)

        return loss
