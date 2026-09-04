from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn as nn

from cauchylift.common import matrixize


# Peak theoretical BF16 TFLOPS for AMD Instinct MI300X (dense, non-sparse)
MI300X_BF16_PEAK_TFLOPS = 1307.4


def estimate_stable_rank(matrix: torch.Tensor, power_iters: int = 3) -> float:
    """Estimate stable rank ||M||_F^2 / ||M||_op^2 via power iteration."""
    if matrix.ndim != 2 or min(matrix.shape) <= 1:
        return 1.0

    m, n = matrix.shape
    frob_sq = float((matrix.to(torch.float32) ** 2).sum().item())
    if frob_sq <= 1e-12:
        return 1.0

    # Power iteration for spectral norm approximation
    v = torch.randn(n, 1, device=matrix.device, dtype=matrix.dtype)
    v = v / (torch.linalg.vector_norm(v) + 1e-8)

    for _ in range(power_iters):
        u = torch.mm(matrix, v)
        u = u / (torch.linalg.vector_norm(u) + 1e-8)
        v = torch.mm(matrix.T, u)
        v = v / (torch.linalg.vector_norm(v) + 1e-8)

    sigma_max = float(torch.linalg.vector_norm(torch.mm(matrix, v)).item())
    if sigma_max <= 1e-12:
        return 1.0

    return frob_sq / (sigma_max ** 2)


def compute_gradient_and_update_metrics(
    model: nn.Module,
    param_copies_before: dict[int, torch.Tensor],
    lr: float,
) -> dict[str, Any]:
    """Compute gradient/update cosine, concentration, support, and stable rank."""
    total_dot = 0.0
    total_g_norm_sq = 0.0
    total_u_norm_sq = 0.0

    rep_matrix_grad = None
    rep_matrix_update = None

    for p in model.parameters():
        if p.grad is None or id(p) not in param_copies_before:
            continue

        g = p.grad.detach().to(torch.float32)
        # Update is the difference: delta = (p_before - p_after) / lr
        p_before = param_copies_before[id(p)].to(torch.float32)
        u = (p_before - p.detach().to(torch.float32)) / (lr + 1e-12)

        g_flat = g.reshape(-1)
        u_flat = u.reshape(-1)

        total_dot += float(torch.dot(g_flat, u_flat).item())
        total_g_norm_sq += float((g_flat ** 2).sum().item())
        total_u_norm_sq += float((u_flat ** 2).sum().item())

        # Select a representative 2D weight matrix (e.g. dimension >= 64)
        if rep_matrix_grad is None and g.ndim == 2 and min(g.shape) >= 32:
            rep_matrix_grad = g
            rep_matrix_update = u

    # Cosine similarity
    g_norm = math.sqrt(total_g_norm_sq)
    u_norm = math.sqrt(total_u_norm_sq)
    if g_norm > 1e-12 and u_norm > 1e-12:
        cosine = total_dot / (g_norm * u_norm)
    else:
        cosine = 1.0

    # Concentration and support on representative matrix
    if rep_matrix_grad is not None:
        g_mat = rep_matrix_grad
        m, n = g_mat.shape
        g_sq = g_mat ** 2
        S = float(g_sq.sum().item())

        if S > 1e-12:
            r = g_sq.sum(dim=1)
            c = g_sq.sum(dim=0)
            row_conc = float((r.max() / S).item())
            col_conc = float((c.max() / S).item())
            # Effective support: S^2 / sum(G_{ij}^4) / (m * n)
            g_4th = float((g_sq ** 2).sum().item())
            effective_support = (S ** 2 / (g_4th + 1e-12)) / (m * n)
            # Cotransverse denominators: E_{ij} = 2*S - r_i - c_j
            E = 2.0 * S - r.unsqueeze(1) - c.unsqueeze(0)
            min_denom = float(E[E > 0].min().item()) if (E > 0).any() else 0.0
            max_denom = float(E.max().item())
            boundary_freq = int((E <= 1e-6 * S).sum().item())
        else:
            row_conc = 1.0 / m
            col_conc = 1.0 / n
            effective_support = 1.0
            min_denom = 0.0
            max_denom = 0.0
            boundary_freq = 0

        # Stable rank of update
        assert rep_matrix_update is not None
        stable_rank = estimate_stable_rank(rep_matrix_update)
    else:
        row_conc = 0.0
        col_conc = 0.0
        effective_support = 1.0
        min_denom = 0.0
        max_denom = 0.0
        boundary_freq = 0
        stable_rank = 1.0

    return {
        "grad_update_cosine": float(cosine),
        "row_concentration": float(row_conc),
        "col_concentration": float(col_conc),
        "effective_support": float(effective_support),
        "update_stable_rank": float(stable_rank),
        "min_denominator": float(min_denom),
        "max_denominator": float(max_denom),
        "boundary_frequency": int(boundary_freq),
    }


@dataclass
class StepRecord:
    step: int
    loss: float
    val_loss: float | None
    lr: float
    tokens: int
    wall_time: float
    step_time: float
    opt_time: float
    throughput_tok_per_sec: float
    peak_memory_bytes: int
    grad_update_cosine: float
    row_concentration: float
    col_concentration: float
    effective_support: float
    update_stable_rank: float
    min_denominator: float
    max_denominator: float
    boundary_frequency: int
    loss_spike: bool
    mfu_estimate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MetricsLogger:
    """Structured local JSONL logger satisfying Phase 4 research contract."""

    def __init__(self, log_path: str) -> None:
        self.log_path = log_path
        os.makedirs(os.path.dirname(os.path.abspath(log_path)), exist_ok=True)
        self.records: list[StepRecord] = []
        self.running_loss_avg: float | None = None

        # Reset log file
        with open(self.log_path, "w", encoding="utf-8") as f:
            pass

    def check_loss_spike(self, loss: float, threshold_factor: float = 1.5) -> bool:
        if self.running_loss_avg is None:
            self.running_loss_avg = loss
            return False
        is_spike = loss > threshold_factor * self.running_loss_avg
        # Update running average with momentum 0.95
        self.running_loss_avg = 0.95 * self.running_loss_avg + 0.05 * loss
        return is_spike

    def log_step(self, record: StepRecord) -> None:
        self.records.append(record)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict()) + "\n")
