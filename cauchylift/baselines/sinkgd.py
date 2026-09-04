from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer


def sinkhorn_normalize(G: torch.Tensor, rounds: int = 5, eps: float = 1e-8) -> torch.Tensor:
    """Sinkhorn alternating L2 row and column normalization for 5 rounds."""
    assert G.ndim == 2, f"Expected 2D matrix, got {G.ndim}D"
    X = G.to(torch.float32)

    for _ in range(rounds):
        # Row normalization
        row_norms = torch.linalg.vector_norm(X, dim=1, keepdim=True) + eps
        X = X / row_norms
        # Column normalization
        col_norms = torch.linalg.vector_norm(X, dim=0, keepdim=True) + eps
        X = X / col_norms

    # Scale to Frobenius norm sqrt(min(m, n))
    m, n = G.shape
    target_norm = math.sqrt(min(m, n))
    frob_norm = torch.linalg.vector_norm(X) + eps
    X = X * (target_norm / frob_norm)

    return X.to(dtype=G.dtype)


class SinkGD(Optimizer):
    """Sinkhorn Gradient Descent (SinkGD) with 5 normalization rounds.

    Stateless optimizer that applies alternating row and column L2 normalization
    to 2D matrices, balancing row and column marginals.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        rounds: int = 5,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            rounds=rounds,
            eps=eps,
            weight_decay=weight_decay,
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
            rounds = group["rounds"]
            eps = group["eps"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                if p.ndim == 2 and min(p.shape) > 1:
                    # 2D matrix Sinkhorn update
                    update = sinkhorn_normalize(grad, rounds=rounds, eps=eps)
                else:
                    # 1D or vector parameter: normalized gradient update
                    g_fp32 = grad.to(torch.float32)
                    norm = torch.linalg.vector_norm(g_fp32) + eps
                    update = (g_fp32 / norm).to(p.dtype)

                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(update, alpha=-lr)

        return loss
