from __future__ import annotations

import math
import pytest
import torch

from cauchylift.baselines import (
    AdamW,
    Muon,
    NormalizedGD,
    SOAP,
    SignDescent,
    SinkGD,
    create_optimizer,
)
from cauchylift.baselines.muon import zeropower_via_newtonschulz5
from cauchylift.baselines.sinkgd import sinkhorn_normalize


def test_muon_newton_schulz_properties():
    """Verify that Newton-Schulz 5 produces an approximately orthogonal matrix."""
    torch.manual_seed(42)
    G = torch.randn(32, 32)
    X = zeropower_via_newtonschulz5(G, steps=5)

    # For an orthogonal matrix, X @ X.T should be close to identity
    I = torch.eye(32)
    Gram = torch.mm(X, X.T)
    diff = (Gram - I).abs().max().item()
    assert diff < 0.5, f"Newton-Schulz Gram matrix deviated from identity: {diff}"


def test_sinkgd_normalization_properties():
    """Verify that 5-round Sinkhorn normalization balances row and column norms."""
    torch.manual_seed(42)
    G = torch.randn(16, 16)
    X = sinkhorn_normalize(G, rounds=5)

    # Check that Frobenius norm equals sqrt(min(m, n)) = 4.0
    frob_norm = float(torch.linalg.vector_norm(X).item())
    assert abs(frob_norm - 4.0) < 1e-4, f"SinkGD norm was {frob_norm}, expected 4.0"

    # Check row and col norms are roughly balanced
    row_norms = torch.linalg.vector_norm(X, dim=1)
    col_norms = torch.linalg.vector_norm(X, dim=0)
    assert (row_norms.max() - row_norms.min()) < 0.1
    assert (col_norms.max() - col_norms.min()) < 0.1


def test_normalized_gd_and_sign_descent_equations():
    """Verify that NormalizedGD and SignDescent obey their explicit definitions."""
    p1 = torch.nn.Parameter(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))  # norm = 5.0, radius = sqrt(2)
    p1.grad = p1.clone()
    opt_ngd = NormalizedGD([p1], lr=0.1)
    opt_ngd.step()
    # Expected update: lr * sqrt(2) * G / ||G||_F = 0.1 * 1.41421356 * [[0.6, 0.8], [0, 0]]
    expected_step = 0.1 * math.sqrt(2) * (3.0 / 5.0)
    actual_step = 3.0 - p1[0, 0].item()
    assert abs(actual_step - expected_step) < 1e-5

    p2 = torch.nn.Parameter(torch.tensor([[3.0, -4.0]]))  # shape [1, 2], radius = 1, signs = [[1, -1]]
    p2.grad = torch.tensor([[5.0, -10.0]])
    opt_sd = SignDescent([p2], lr=0.1)
    opt_sd.step()
    # Sign is [1, -1], norm is sqrt(2), radius is sqrt(max(1, 2)) = sqrt(2) -> direction is [1.0, -1.0]
    expected_step2 = 0.1 * math.sqrt(2) * (1.0 / math.sqrt(2))
    actual_step2 = 3.0 - p2[0, 0].item()
    assert abs(actual_step2 - expected_step2) < 1e-5


def test_all_optimizers_step_on_toy_model():
    """All baselines and CauchyLift must execute valid step without NaN/Inf."""
    optimizers = [
        "cauchylift",
        "adamw",
        "muon",
        "soap",
        "sinkgd",
        "normalized_gd",
        "sign_descent",
    ]

    for opt_name in optimizers:
        torch.manual_seed(999)
        model = torch.nn.Sequential(
            torch.nn.Linear(8, 16, bias=False),
            torch.nn.Linear(16, 4, bias=False),
        )
        opt = create_optimizer(opt_name, model, lr=1e-3)
        x = torch.randn(2, 8)
        y = model(x).sum()
        y.backward()
        opt.step()
        opt.zero_grad()

        for p in model.parameters():
            assert torch.isfinite(p).all(), f"Optimizer {opt_name} produced non-finite parameter values"


def test_adamw_matches_torch_adamw():
    """Verify AdamW baseline numerically matches PyTorch AdamW across steps."""
    torch.manual_seed(42)
    p_baseline = torch.randn(16, 32)
    p_torch = p_baseline.clone()

    p_baseline = torch.nn.Parameter(p_baseline)
    p_torch = torch.nn.Parameter(p_torch)

    lr = 1e-3
    betas = (0.9, 0.95)
    eps = 1e-8
    wd = 0.01

    opt_baseline = AdamW([p_baseline], lr=lr, betas=betas, eps=eps, weight_decay=wd, backend="reference")
    opt_torch = torch.optim.AdamW([p_torch], lr=lr, betas=betas, eps=eps, weight_decay=wd)

    for _ in range(10):
        grad = torch.randn(16, 32)
        p_baseline.grad = grad.clone()
        p_torch.grad = grad.clone()

        opt_baseline.step()
        opt_torch.step()

        torch.testing.assert_close(p_baseline, p_torch, atol=1e-6, rtol=1e-5)


def test_muon_with_custom_param_groups():
    """Verify Muon handles custom param_groups containing both 2D and 1D params without error."""
    torch.manual_seed(123)
    p2d = torch.nn.Parameter(torch.randn(32, 64))
    p1d = torch.nn.Parameter(torch.randn(64))

    param_groups = [
        {"params": [p2d], "weight_decay": 0.01},
        {"params": [p1d], "weight_decay": 0.0},
    ]

    opt = Muon(param_groups, lr=0.02, adamw_lr=6e-4, backend="reference")
    p2d.grad = torch.randn_like(p2d)
    p1d.grad = torch.randn_like(p1d)

    opt.step()
    assert torch.isfinite(p2d).all()
    assert torch.isfinite(p1d).all()


def test_muon_adamw_learning_rate_separation():
    """Verify Muon applies adamw_lr to 1D params and lr to 2D params."""
    torch.manual_seed(456)
    p2d = torch.nn.Parameter(torch.zeros(16, 16))
    p1d = torch.nn.Parameter(torch.zeros(16))

    opt = Muon([p2d, p1d], lr=0.02, adamw_lr=6e-4, weight_decay=0.0, backend="reference")
    p2d.grad = torch.ones_like(p2d)
    p1d.grad = torch.ones_like(p1d)

    opt.step()
    # 1D step with unit gradient under AdamW with initial buffer produces exactly adamw_lr step
    delta_1d = p1d.abs().max().item()
    assert math.isclose(delta_1d, 6e-4, rel_tol=1e-4), f"Expected delta_1d close to 6e-4, got {delta_1d}"
    assert opt.param_groups[0]["lr"] == 0.02
    assert opt.param_groups[1]["lr"] == 6e-4

