from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class SOAP(Optimizer):
    """SOAP optimizer (Shampoo with Adam in Preconditioner eigenbasis).

    References:
        Vyas et al., 'SOAP: Improving and Stabilizing Shampoo using Adam', ICML 2024.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.95),
        shampoo_beta: float = 0.95,
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        precondition_frequency: int = 10,
        max_precondition_dim: int = 2048,
    ) -> None:
        defaults = dict(
            lr=lr,
            betas=betas,
            shampoo_beta=shampoo_beta,
            eps=eps,
            weight_decay=weight_decay,
            precondition_frequency=precondition_frequency,
            max_precondition_dim=max_precondition_dim,
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
            beta1, beta2 = group["betas"]
            shampoo_beta = group["shampoo_beta"]
            eps = group["eps"]
            wd = group["weight_decay"]
            freq = group["precondition_frequency"]
            max_dim = group["max_precondition_dim"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "step" not in state:
                    state["step"] = 0

                state["step"] += 1
                step = state["step"]

                # Check if parameter qualifies for 2D SOAP preconditioning
                is_2d_matrix = (
                    p.ndim == 2
                    and p.shape[0] <= max_dim
                    and p.shape[1] <= max_dim
                    and min(p.shape) > 1
                )

                if is_2d_matrix:
                    m, n = p.shape
                    g_fp32 = grad.to(torch.float32)

                    if "L" not in state:
                        state["L"] = torch.eye(m, device=p.device, dtype=torch.float32) * 1e-4
                        state["R"] = torch.eye(n, device=p.device, dtype=torch.float32) * 1e-4
                        state["Q_L"] = torch.eye(m, device=p.device, dtype=torch.float32)
                        state["Q_R"] = torch.eye(n, device=p.device, dtype=torch.float32)
                        state["exp_avg"] = torch.zeros(m, n, device=p.device, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros(m, n, device=p.device, dtype=torch.float32)

                    L = state["L"]
                    R = state["R"]
                    Q_L = state["Q_L"]
                    Q_R = state["Q_R"]

                    # Update covariance matrices
                    L.mul_(shampoo_beta).addmm_(g_fp32, g_fp32.T, alpha=(1.0 - shampoo_beta) / n)
                    R.mul_(shampoo_beta).addmm_(g_fp32.T, g_fp32, alpha=(1.0 - shampoo_beta) / m)

                    # Update eigenbases every `precondition_frequency` steps
                    if step == 1 or step % freq == 0:
                        try:
                            # Symmetrize before eigh
                            _, Q_L_new = torch.linalg.eigh(0.5 * (L + L.T))
                            _, Q_R_new = torch.linalg.eigh(0.5 * (R + R.T))
                            Q_L.copy_(Q_L_new)
                            Q_R.copy_(Q_R_new)
                        except Exception:
                            # Fallback if eigh encounters numerical issue
                            pass

                    # Project gradient into eigenbasis
                    # G_rot = Q_L^T @ G @ Q_R
                    g_rot = torch.mm(torch.mm(Q_L.T, g_fp32), Q_R)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg.mul_(beta1).add_(g_rot, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_rot, g_rot, value=1.0 - beta2)

                    bias_corr1 = 1.0 - beta1 ** step
                    bias_corr2 = 1.0 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                    u_rot = (exp_avg / bias_corr1) / denom

                    # Project back to parameter space: U = Q_L @ U_rot @ Q_R^T
                    u = torch.mm(torch.mm(Q_L, u_rot), Q_R.T)

                    # Decoupled weight decay
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    p.add_(u.to(p.dtype), alpha=-lr)

                else:
                    # AdamW fallback for 1D vectors and large embedding tables
                    if "exp_avg" not in state:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]
                    g_fp32 = grad.to(torch.float32)

                    exp_avg.mul_(beta1).add_(g_fp32, alpha=1.0 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1.0 - beta2)

                    bias_corr1 = 1.0 - beta1 ** step
                    bias_corr2 = 1.0 - beta2 ** step

                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_corr2)).add_(eps)
                    step_size = lr / bias_corr1

                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)

                    p.addcdiv_(exp_avg.to(p.dtype), denom.to(p.dtype), value=-step_size)

        return loss
