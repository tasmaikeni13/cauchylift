"""CauchyLift: Curvature-Adaptive Matrix Optimizer with Parameter Routing and Hardware-Fused Acceleration."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch

from .reference import cauchylift_reference_step
from .xla import (
    adamw_xla_foreach_step_,
    cauchylift_xla_foreach_step_,
    cauchylift_xla_step_,
    is_tpu_available,
)


def is_2d_hidden_matrix(param: torch.Tensor) -> bool:
    """Determine if parameter is a dense 2D internal linear transformation operator."""
    return param.ndim == 2 and min(param.shape) > 1 and max(param.shape) < 10000


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
                    raise ValueError("A tied parameter cannot have conflicting learning rates")
                continue
            seen_lr[key] = lr
            unique.append(parameter)
        group["params"] = unique
        groups.append(group)
    return groups


class CauchyLift(torch.optim.Optimizer):
    """Canonical Curvature-Adaptive Matrix Optimizer for Deep Learning and Pretraining.

    CauchyLift combines:
    1. Canonical Parameter Routing:
       - 2D Hidden Linear Matrices: Updated via core CauchyLift (historical momentum
         low-pass filter, Additive Fiber RMS curvature denominator, longest-fiber
         Frobenius sphere projection, decoupled weight decay).
       - 1D Parameters & Embeddings: Automatically routed to coordinate-wise AdamW
         updates (RMSNorm/LayerNorm scales, biases, token lookup and head tables).
         This is the default, native behavior of CauchyLift.
    2. Additive Fiber RMS Cauchy lifting:
           D_{ij} = RMS(M_{i,:}) + RMS(M_{:,j})
           Z_{ij} = M_{ij} / D_{ij}
           U = sqrt(max(m, n)) * Z / ||Z||_F
       providing Riemannian curvature adaptation with O(N^2) complexity.
    3. Decoupled weight decay:
           W_{t+1} = W_t * (1 - lr * weight_decay) - lr * U
       preventing Frobenius norm runaway and maintaining optimal layer conditioning.
    4. Hardware-Fused Multi-Tensor TPU Execution:
       Native Google Cloud TPU v4 (Torch-XLA / PJRT) systolic execution with FP32
       vector register accumulation for reductions, BF16 MXU matrix ops, and zero
       host-device synchronization overhead inside training step loops.
    """

    def __init__(
        self,
        params: Iterable[Any],
        lr: float = 1e-3,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        *,
        adamw_lr: float = 6e-4,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_weight_decay: float = 0.0,
        nesterov: bool = False,
        backend: str = "auto",
        canonical_routing: bool = True,
        strict: bool = True,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum parameter: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay parameter: {weight_decay}")
        if backend not in {"auto", "reference", "xla", "tpu"}:
            raise ValueError("Backend must be 'auto', 'reference', 'xla', or 'tpu'")

        self.backend = backend.lower()
        self.strict = strict
        self.canonical_routing = canonical_routing

        param_list = list(params)
        if len(param_list) > 0 and isinstance(param_list[0], dict):
            # Param groups provided
            if canonical_routing:
                expanded_groups = []
                for grp in param_list:
                    if grp.get("is_cauchylift") is not None:
                        expanded_groups.append(grp)
                        continue
                    cl_p = []
                    adam_p = []
                    for p in grp["params"]:
                        if is_2d_hidden_matrix(p):
                            cl_p.append(p)
                        else:
                            adam_p.append(p)
                    if cl_p and adam_p:
                        g_cl = dict(grp)
                        g_cl["params"] = cl_p
                        g_cl["is_cauchylift"] = True
                        g_cl.setdefault("base_lr", g_cl.get("lr", lr))
                        expanded_groups.append(g_cl)

                        g_adam = dict(grp)
                        g_adam["params"] = adam_p
                        g_adam["lr"] = grp.get("adamw_lr", adamw_lr)
                        g_adam["betas"] = grp.get("adamw_betas", adamw_betas)
                        g_adam["eps"] = grp.get("adamw_eps", adamw_eps)
                        g_adam["weight_decay"] = grp.get("adamw_weight_decay", grp.get("weight_decay", adamw_weight_decay))
                        g_adam["is_cauchylift"] = False
                        g_adam.setdefault("base_lr", g_adam["lr"])
                        expanded_groups.append(g_adam)
                    elif cl_p:
                        g_cl = dict(grp)
                        g_cl["is_cauchylift"] = True
                        g_cl.setdefault("base_lr", g_cl.get("lr", lr))
                        expanded_groups.append(g_cl)
                    elif adam_p:
                        g_adam = dict(grp)
                        g_adam.setdefault("lr", adamw_lr)
                        g_adam["is_cauchylift"] = False
                        g_adam.setdefault("base_lr", g_adam["lr"])
                        expanded_groups.append(g_adam)
                    else:
                        expanded_groups.append(grp)
                processed_params = _deduplicate_parameters(expanded_groups, lr)
            else:
                processed_params = _deduplicate_parameters(param_list, lr)
        else:
            # Single flat parameter list
            if canonical_routing:
                cl_params = []
                adamw_params = []
                for p in param_list:
                    if is_2d_hidden_matrix(p):
                        cl_params.append(p)
                    else:
                        adamw_params.append(p)

                groups = []
                if cl_params:
                    groups.append({
                        "params": cl_params,
                        "lr": lr,
                        "momentum": momentum,
                        "weight_decay": weight_decay,
                        "nesterov": nesterov,
                        "is_cauchylift": True,
                        "base_lr": lr,
                    })
                if adamw_params:
                    groups.append({
                        "params": adamw_params,
                        "lr": adamw_lr,
                        "betas": adamw_betas,
                        "eps": adamw_eps,
                        "weight_decay": adamw_weight_decay,
                        "is_cauchylift": False,
                        "base_lr": adamw_lr,
                    })
                processed_params = _deduplicate_parameters(groups, lr)
            else:
                processed_params = _deduplicate_parameters(param_list, lr)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=nesterov,
            adamw_lr=adamw_lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_weight_decay=adamw_weight_decay,
            canonical_routing=canonical_routing,
        )
        super().__init__(processed_params, defaults)

    def _is_xla_active(self) -> bool:
        if self.backend in ("xla", "tpu"):
            return True
        if self.backend == "reference":
            return False
        for group in self.param_groups:
            for p in group["params"]:
                if p.device.type == "xla":
                    return True
        return is_tpu_available()

    @torch.no_grad()
    def step(self, closure: Any = None) -> Any:
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        use_xla = self._is_xla_active()

        for group in self.param_groups:
            group_is_cl = group.get("is_cauchylift", None)
            lr_cl = float(group["lr"])
            momentum = float(group.get("momentum", self.defaults.get("momentum", 0.95)))
            weight_decay = float(group.get("weight_decay", self.defaults.get("weight_decay", 0.01)))
            nesterov = bool(group.get("nesterov", self.defaults.get("nesterov", False)))

            adamw_lr_val = float(group.get("adamw_lr", lr_cl if group_is_cl is False else self.defaults.get("adamw_lr", 6e-4)))
            beta1, beta2 = group.get("adamw_betas", group.get("betas", self.defaults.get("adamw_betas", (0.9, 0.95))))
            eps = float(group.get("adamw_eps", group.get("eps", self.defaults.get("adamw_eps", 1e-8))))
            adamw_wd = float(group.get("adamw_weight_decay", weight_decay if group_is_cl is False else self.defaults.get("adamw_weight_decay", 0.0)))

            xla_cl_p: list[torch.Tensor] = []
            xla_cl_g: list[torch.Tensor] = []
            xla_cl_m: list[torch.Tensor] = []

            xla_adamw_p: list[torch.Tensor] = []
            xla_adamw_g: list[torch.Tensor] = []
            xla_adamw_ea: list[torch.Tensor] = []
            xla_adamw_eas: list[torch.Tensor] = []
            adamw_max_step = 0

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if gradient.layout != torch.strided:
                    gradient = gradient.to_dense()

                state = self.state[parameter]

                if group_is_cl is True:
                    use_cl = True
                elif group_is_cl is False:
                    use_cl = False
                else:
                    use_cl = is_2d_hidden_matrix(parameter) if self.canonical_routing else True

                if use_cl:
                    # ================= Core CauchyLift 2D Update =================
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(gradient)
                    momentum_buffer = state["momentum_buffer"]

                    if use_xla and parameter.device.type == "xla":
                        xla_cl_p.append(parameter)
                        xla_cl_g.append(gradient)
                        xla_cl_m.append(momentum_buffer)
                    else:
                        cauchylift_reference_step(
                            parameter,
                            gradient,
                            momentum_buffer,
                            lr_cl,
                            momentum=momentum,
                            weight_decay=weight_decay,
                            nesterov=nesterov,
                        )
                else:
                    # ================= Coordinate-wise AdamW Update =================
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)

                    state["step"] += 1
                    step_count = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    if use_xla and parameter.device.type == "xla":
                        xla_adamw_p.append(parameter)
                        xla_adamw_g.append(gradient)
                        xla_adamw_ea.append(exp_avg)
                        xla_adamw_eas.append(exp_avg_sq)
                        adamw_max_step = max(adamw_max_step, step_count)
                    else:
                        g_fp32 = gradient.to(torch.float32)
                        exp_avg.mul_(beta1).add_(g_fp32, alpha=1.0 - beta1)
                        exp_avg_sq.mul_(beta2).addcmul_(g_fp32, g_fp32, value=1.0 - beta2)

                        bias_correction1 = 1.0 - beta1 ** step_count
                        bias_correction2 = 1.0 - beta2 ** step_count

                        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                        step_size = adamw_lr_val / bias_correction1

                        if adamw_wd != 0.0:
                            parameter.mul_(1.0 - adamw_lr_val * adamw_wd)

                        update = exp_avg / denom
                        parameter.add_(update.to(parameter.dtype), alpha=-step_size)

            # Batch multi-tensor execution for eligible parameter tensors on TPU
            if xla_cl_p:
                cauchylift_xla_foreach_step_(
                    xla_cl_p,
                    xla_cl_g,
                    xla_cl_m,
                    learning_rate=lr_cl,
                    momentum=momentum,
                    weight_decay=weight_decay,
                    nesterov=nesterov,
                    accumulation_dtype=torch.float32,
                    mark_step=False,
                )
            if xla_adamw_p:
                adamw_xla_foreach_step_(
                    xla_adamw_p,
                    xla_adamw_g,
                    xla_adamw_ea,
                    xla_adamw_eas,
                    step=adamw_max_step,
                    learning_rate=adamw_lr_val,
                    beta1=beta1,
                    beta2=beta2,
                    eps=eps,
                    weight_decay=adamw_wd,
                    mark_step=False,
                )

        return loss

    def persistent_tensor_summary(self) -> dict[str, int]:
        """Return counts and bytes of persistent optimizer state."""
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
