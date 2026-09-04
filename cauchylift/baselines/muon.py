from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Newton-Schulz quintic iteration for matrix orthogonalization (Muon).

    Coefficients: a = 3.4445, b = -4.7750, c = 2.0315.
    Approximates the nearest orthogonal matrix (polar factor) to G.
    """
    assert G.ndim == 2, f"Expected 2D tensor, got {G.ndim}D"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.dtype != torch.bfloat16 else G
    X = X / (X.norm() + eps)

    m, n = X.shape
    transposed = False
    if m > n:
        X = X.T
        m, n = n, m
        transposed = True

    for _ in range(steps):
        # A = X @ X.T has shape [m, m]
        A = torch.mm(X, X.T)
        B = b * A + c * torch.mm(A, A)
        X = a * X + torch.mm(B, X)

    if transposed:
        X = X.T

    return X.to(dtype=G.dtype)


class Muon(Optimizer):
    """Muon optimizer (MomentUm Orthogonalized by Newton-Schulz).

    Applies Newton-Schulz orthogonalization to 2D internal weight matrices,
    and standard AdamW updates to 1D and embedding parameters.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.0,
        adamw_lr: float = 1e-3,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.01,
    ) -> None:
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
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            adamw_lr = group["adamw_lr"]
            beta1, beta2 = group["adamw_betas"]
            adamw_eps = group["adamw_eps"]
            adamw_wd = group["adamw_weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                # Determine if parameter is an internal 2D matrix suitable for Muon
                # Non-2D parameters or embedding tables (e.g. vocab_size > 5000) use AdamW
                is_internal_matrix = p.ndim == 2 and min(p.shape) > 1 and max(p.shape) < 10000

                if is_internal_matrix:
                    # Muon update path
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)

                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad)

                    update = grad + momentum * buf if group["nesterov"] else buf
                    # Apply Newton-Schulz iteration
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Scale by sqrt(max(1, m/n))
                    m, n = p.shape
                    scale = math.sqrt(max(1.0, m / n))
                    update.mul_(scale)

                    # Decoupled weight decay
                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)

                    p.add_(update, alpha=-lr)

                else:
                    # AdamW fallback for embeddings and 1D normalization gains
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                    state["step"] += 1
                    step = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    g_fp32 = grad.to(torch.float32)
                    exp_avg.mul_(beta1).add_(g_fp32, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1 - beta2)

                    bias_correction1 = 1.0 - beta1 ** step
                    bias_correction2 = 1.0 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(adamw_eps)
                    step_size = adamw_lr / bias_correction1

                    if adamw_wd != 0.0:
                        p.mul_(1.0 - adamw_lr * adamw_wd)

                    p.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)

        return loss
