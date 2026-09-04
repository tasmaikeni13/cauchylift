from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW, Optimizer

from cauchylift.optimizer import CauchyLift
from .controls import NormalizedGD, SignDescent
from .muon import Muon
from .sinkgd import SinkGD
from .soap import SOAP


def create_optimizer(
    name: str,
    model_or_params: nn.Module | Iterable[torch.nn.Parameter],
    lr: float,
    weight_decay: float = 0.0,
    **kwargs: Any,
) -> Optimizer:
    """Create an optimizer by name with a unified interface.

    Supported optimizer names:
    - 'cauchylift' (auto backend, native HIP on GPU, reference on CPU)
    - 'cauchylift_hip' (strict=False native fast path on GPU)
    - 'cauchylift_reference' (FP32 reference path)
    - 'adamw' (standard AdamW, fused=True on GPU if available)
    - 'muon' (Muon with Newton-Schulz 5)
    - 'soap' (SOAP with Shampoo preconditioning in eigenbasis)
    - 'sinkgd' (SinkGD with 5-round Sinkhorn normalization)
    - 'normalized_gd' (Normalized gradient descent mechanism control)
    - 'sign_descent' (Sign descent mechanism control)
    """
    name_clean = name.lower().strip()
    if isinstance(model_or_params, nn.Module):
        params = [p for p in model_or_params.parameters() if p.requires_grad]
    else:
        params = list(model_or_params)

    if name_clean in ("cauchylift", "cauchylift_auto"):
        # CauchyLift does not accept weight decay in primitive
        return CauchyLift(params, lr=lr, backend="auto", strict=True)
    elif name_clean in ("cauchylift_hip", "cauchylift_fast"):
        return CauchyLift(params, lr=lr, backend="hip", strict=False)
    elif name_clean == "cauchylift_reference":
        return CauchyLift(params, lr=lr, backend="reference", strict=True)
    elif name_clean == "adamw":
        betas = kwargs.get("betas", (0.9, 0.95))
        eps = kwargs.get("eps", 1e-8)
        # Attempt fused AdamW if on CUDA/ROCm
        fused = kwargs.get("fused", None)
        if fused is None:
            fused = torch.cuda.is_available() and any(p.is_cuda for p in params)
        try:
            return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=fused)
        except Exception:
            return AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, fused=False)
    elif name_clean == "muon":
        return Muon(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name_clean == "soap":
        return SOAP(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name_clean == "sinkgd":
        return SinkGD(params, lr=lr, weight_decay=weight_decay, **kwargs)
    elif name_clean in ("normalized_gd", "normalizedgd"):
        return NormalizedGD(params, lr=lr, weight_decay=weight_decay)
    elif name_clean in ("sign_descent", "signdescent"):
        return SignDescent(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}. Supported: cauchylift, adamw, muon, soap, sinkgd, normalized_gd, sign_descent")
