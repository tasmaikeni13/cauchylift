"""AdamW optimizer implementation with Google Cloud TPU / Torch-XLA native kernel dispatch."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch.optim import Optimizer

from cauchylift.xla import (
    adamw_xla_foreach_step_,
    adamw_xla_step_,
    is_tpu_available,
)


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay and TPU/XLA native kernel execution.

    Implements the AdamW algorithm (Loshchilov & Hutter, 2019) with:
    - First moment exponential moving average (beta1)
    - Second moment exponential moving average of squared gradients (beta2)
    - Analytical bias correction (1 - beta1^t, 1 - beta2^t)
    - Decoupled weight decay: W_{t+1} = W_t * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
    - Hardware-native Google Cloud TPU v4 XLA kernel dispatch with multi-tensor foreach batching.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        *,
        backend: str = "auto",
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 parameter: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 parameter: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if backend not in {"auto", "reference", "xla", "tpu"}:
            raise ValueError("Backend must be 'auto', 'reference', 'xla', or 'tpu'")

        self.backend = backend.lower()
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

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
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            xla_params: list[torch.Tensor] = []
            xla_grads: list[torch.Tensor] = []
            xla_exp_avgs: list[torch.Tensor] = []
            xla_exp_avg_sqs: list[torch.Tensor] = []
            max_step = 0

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.layout != torch.strided:
                    grad = grad.to_dense()

                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)

                state["step"] += 1
                step_count = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if use_xla and p.device.type == "xla":
                    xla_params.append(p)
                    xla_grads.append(grad)
                    xla_exp_avgs.append(exp_avg)
                    xla_exp_avg_sqs.append(exp_avg_sq)
                    max_step = max(max_step, step_count)
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

                    update = exp_avg / denom
                    p.add_(update.to(p.dtype), alpha=-step_size)

            if xla_params:
                adamw_xla_foreach_step_(
                    xla_params,
                    xla_grads,
                    xla_exp_avgs,
                    xla_exp_avg_sqs,
                    step=max_step,
                    learning_rate=lr,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=weight_decay,
                )

        return loss
