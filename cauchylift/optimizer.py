from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch

from .hip import cauchylift_hip_step_, is_rocm_available
from .reference import cauchylift_reference


def _deduplicate_parameters(params: Iterable[Any], default_lr: float) -> list[Any]:
    materialized = list(params)
    if not materialized:
        return materialized
    if not isinstance(materialized[0], dict):
        seen: set[int] = set()
        result = []
        for parameter in materialized:
            if id(parameter) not in seen:
                seen.add(id(parameter))
                result.append(parameter)
        return result

    seen_lr: dict[int, float] = {}
    groups = []
    for original in materialized:
        group = dict(original)
        group_params = list(group["params"])
        lr = float(group.get("lr", default_lr))
        unique = []
        for parameter in group_params:
            key = id(parameter)
            if key in seen_lr:
                if seen_lr[key] != lr:
                    raise ValueError("a tied parameter cannot have conflicting learning rates")
                continue
            seen_lr[key] = lr
            unique.append(parameter)
        group["params"] = unique
        groups.append(group)
    return groups


class CauchyLift(torch.optim.Optimizer):
    """State-free optimizer implementing the frozen CauchyLift v0.2 primitive.

    No momentum, moments, weight decay, clipping, epsilon, or fallback optimizer
    is accepted. ``backend='auto'`` selects native HIP only on ROCm tensors.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        *,
        backend: str = "auto",
        strict: bool = True,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if backend not in {"auto", "reference", "hip"}:
            raise ValueError("backend must be 'auto', 'reference', or 'hip'")
        self.backend = backend
        self.strict = strict
        params = _deduplicate_parameters(params, lr)
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.layout != torch.strided:
                    gradient = gradient.to_dense()
                use_hip = self.backend == "hip" or (
                    self.backend == "auto"
                    and parameter.is_cuda
                    and is_rocm_available()
                    and parameter.dtype in (torch.float32, torch.bfloat16)
                )
                if use_hip:
                    cauchylift_hip_step_(
                        parameter,
                        gradient,
                        learning_rate,
                        strict=self.strict,
                    )
                else:
                    direction = cauchylift_reference(gradient)
                    parameter.add_(direction.to(parameter.dtype), alpha=-learning_rate)
        return loss

    def persistent_tensor_summary(self) -> dict[str, int]:
        tensor_count = sum(
            int(torch.is_tensor(value))
            for state in self.state.values()
            for value in state.values()
        )
        tensor_bytes = sum(
            value.numel() * value.element_size()
            for state in self.state.values()
            for value in state.values()
            if torch.is_tensor(value)
        )
        return {"tensors": tensor_count, "bytes": tensor_bytes}
