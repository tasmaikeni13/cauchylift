from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

from cauchylift.common import matrixize, radius_for


class NormalizedGD(Optimizer):
    """Normalized Gradient Descent mechanism control.

    Updates parameters along the unit Frobenius gradient direction scaled by
    rho(m, n) = sqrt(min(m, n)).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                radius = radius_for(p)
                g_fp32 = grad.to(torch.float32)
                frob_norm = torch.linalg.vector_norm(g_fp32)
                if frob_norm > eps:
                    direction = (g_fp32 / frob_norm) * radius
                else:
                    direction = torch.zeros_like(g_fp32)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(direction.to(p.dtype), alpha=-lr)

        return loss


class SignDescent(Optimizer):
    """Sign Descent mechanism control.

    Updates parameters along the sign of the gradient, normalized to
    Frobenius norm rho(m, n) = sqrt(min(m, n)).
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        eps: float = 1e-12,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(lr=lr, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                radius = radius_for(p)
                sign_g = torch.sign(grad).to(torch.float32)
                norm = torch.linalg.vector_norm(sign_g)
                if norm > eps:
                    direction = (sign_g / norm) * radius
                else:
                    direction = torch.zeros_like(sign_g)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(direction.to(p.dtype), alpha=-lr)

        return loss
