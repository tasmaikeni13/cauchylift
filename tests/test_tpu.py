"""Google Cloud TPU / XLA tests for CauchyLift."""

from __future__ import annotations

import pytest
import torch

from cauchylift import CauchyLift, cauchylift_reference_step
from cauchylift.xla import (
    adamw_xla_foreach_step_,
    adamw_xla_step_,
    cauchylift_direction_xla,
    cauchylift_xla_foreach_step_,
    cauchylift_xla_step_,
    get_hlo_text,
    get_tpu_device,
    is_tpu_available,
    muon_newton_schulz_xla,
    muon_xla_foreach_step_,
    muon_xla_step_,
    sync_tpu,
)

pytestmark = pytest.mark.skipif(not is_tpu_available(), reason="Google Cloud TPU not available")


def test_tpu_is_available_and_detected():
    """Verify that Google Cloud TPU is available and devices are initialized."""
    assert is_tpu_available()
    dev = get_tpu_device()
    assert dev.type == "xla"
    import torch_xla.core.xla_model as xm
    tpu_devices = xm.get_xla_supported_devices()
    assert len(tpu_devices) > 0, "No TPU devices found"


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (64, 128), (256, 256)])
def test_tpu_step_matches_reference(dtype, shape):
    """Verify single-tensor TPU/XLA step matches the PyTorch reference step within numerical tolerance."""
    torch.manual_seed(42)
    dev = get_tpu_device()

    p_ref = torch.randn(shape, dtype=dtype)
    g = torch.randn(shape, dtype=dtype)
    m_ref = torch.zeros_like(p_ref)

    p_tpu = p_ref.clone().to(dev)
    g_tpu = g.clone().to(dev)
    m_tpu = m_ref.clone().to(dev)

    lr = 1e-3
    momentum = 0.95
    weight_decay = 0.01

    cauchylift_reference_step(
        p_ref, g, m_ref, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    cauchylift_xla_step_(
        p_tpu, g_tpu, m_tpu, learning_rate=lr, momentum=momentum, weight_decay=weight_decay
    )
    sync_tpu()

    atol = 2e-3 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-3 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(p_tpu.cpu(), p_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(m_tpu.cpu(), m_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_tpu_foreach_step_matches_single(dtype):
    """Verify multi-tensor foreach TPU step matches single-tensor execution."""
    torch.manual_seed(1337)
    dev = get_tpu_device()
    shapes = [(16, 64), (128, 256), (32, 32)]

    p_single = [torch.randn(s, device=dev, dtype=dtype) for s in shapes]
    p_multi = [p.clone() for p in p_single]
    grads = [torch.randn(s, device=dev, dtype=dtype) for s in shapes]
    m_single = [torch.zeros_like(p) for p in p_single]
    m_multi = [torch.zeros_like(p) for p in p_single]

    lr = 2e-3
    momentum = 0.9
    weight_decay = 0.02

    for p, g, m in zip(p_single, grads, m_single):
        cauchylift_xla_step_(p, g, m, learning_rate=lr, momentum=momentum, weight_decay=weight_decay)

    cauchylift_xla_foreach_step_(
        p_multi, grads, m_multi, learning_rate=lr, momentum=momentum, weight_decay=weight_decay
    )
    sync_tpu()

    atol = 2e-3 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-3 if dtype == torch.bfloat16 else 1e-4
    for ps, pm in zip(p_single, p_multi):
        torch.testing.assert_close(pm.cpu(), ps.cpu(), atol=atol, rtol=rtol)


def test_optimizer_full_step_on_tpu():
    """Verify full CauchyLift optimizer step in backend='auto' on TPU."""
    torch.manual_seed(42)
    dev = get_tpu_device()
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128, bias=False),
        torch.nn.Linear(128, 32, bias=False),
    ).to(device=dev, dtype=torch.bfloat16)

    opt = CauchyLift(model.parameters(), lr=1e-3, momentum=0.95, weight_decay=0.01, backend="auto")
    x = torch.randn(8, 64, device=dev, dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()

    opt.step()
    opt.zero_grad()
    sync_tpu()

    summary = opt.persistent_tensor_summary()
    assert summary["tensors"] == 2
    assert summary["bytes"] > 0


def test_hlo_inspection_on_tpu():
    """Verify that lowered HLO text can be extracted and contains graph operations."""
    dev = get_tpu_device()
    p = torch.randn(32, 64, device=dev)
    g = torch.randn(32, 64, device=dev)
    m = torch.zeros_like(p)
    cauchylift_xla_step_(p, g, m, learning_rate=1e-3)
    hlo = get_hlo_text(p)
    assert len(hlo) > 0
    assert "HloModule" in hlo or "ENTRY" in hlo
    sync_tpu()


def test_muon_newton_schulz_on_tpu():
    """Verify that TPU-optimized Newton-Schulz orthogonalization produces orthogonal matrices on TPU."""
    from cauchylift.xla import muon_newton_schulz_xla
    torch.manual_seed(42)
    dev = get_tpu_device()

    # Test both square and rectangular matrices
    for shape in [(32, 32), (32, 64), (64, 32)]:
        G = torch.randn(shape, device=dev, dtype=torch.bfloat16)
        X = muon_newton_schulz_xla(G, steps=5)
        sync_tpu()

        assert X.shape == shape
        assert X.dtype == torch.bfloat16
        assert torch.isfinite(X).all()

        # For square 32x32, check polar orthogonality (Gram matrix close to scaled identity)
        if shape == (32, 32):
            X_cpu = X.cpu().float()
            Gram = torch.mm(X_cpu, X_cpu.T)
            I = torch.eye(32)
            diff = (Gram - I).abs().max().item()
            assert diff < 0.6, f"Muon TPU Newton-Schulz Gram deviated: {diff}"


def test_muon_xla_step_on_tpu():
    """Verify in-place Muon step execution on TPU with momentum and weight decay."""
    from cauchylift.xla import muon_xla_step_
    torch.manual_seed(123)
    dev = get_tpu_device()

    p = torch.randn(64, 128, device=dev, dtype=torch.bfloat16)
    g = torch.randn(64, 128, device=dev, dtype=torch.bfloat16)
    buf = torch.zeros_like(p)

    p_orig = p.clone()
    muon_xla_step_(p, g, buf, learning_rate=0.02, momentum=0.95, weight_decay=0.01, nesterov=True)
    sync_tpu()

    assert torch.isfinite(p).all()
    assert torch.isfinite(buf).all()
    # Ensure parameter actually updated
    delta = (p - p_orig).abs().max().item()
    assert delta > 1e-4, f"Parameter did not update: delta={delta}"


def test_muon_optimizer_full_step_on_tpu():
    """Verify full Muon optimizer step across 2D weights and 1D norms on TPU."""
    from cauchylift.baselines.muon import Muon
    torch.manual_seed(42)
    dev = get_tpu_device()

    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = torch.nn.Linear(32, 64, bias=False)
            self.linear2 = torch.nn.Linear(64, 32, bias=False)
            self.norm = torch.nn.LayerNorm(32)

        def forward(self, x):
            return self.norm(self.linear2(torch.relu(self.linear1(x))))

    model = TinyBlock().to(device=dev, dtype=torch.bfloat16)
    opt = Muon(model.parameters(), lr=0.02, adamw_lr=1e-3, backend="auto")

    x = torch.randn(4, 32, device=dev, dtype=torch.bfloat16)
    out = model(x)
    loss = out.sum()
    loss.backward()

    opt.step()
    opt.zero_grad()
    sync_tpu()

    for p in model.parameters():
        assert torch.isfinite(p).all(), "Non-finite parameter encountered after Muon step on TPU"


def test_adamw_xla_step_matches_torch_adamw_on_tpu():
    """Verify adamw_xla_step_ on TPU matches PyTorch CPU AdamW reference."""
    torch.manual_seed(42)
    dev = get_tpu_device()

    shape = (32, 64)
    p_ref = torch.randn(shape, dtype=torch.float32)
    p_tpu = p_ref.clone().to(dev)

    opt_ref = torch.optim.AdamW([p_ref], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    ea_tpu = torch.zeros_like(p_tpu)
    eas_tpu = torch.zeros_like(p_tpu)

    for step in range(1, 6):
        grad = torch.randn(shape, dtype=torch.float32)
        p_ref.grad = grad.clone()
        opt_ref.step()

        adamw_xla_step_(
            p_tpu,
            grad.to(dev),
            ea_tpu,
            eas_tpu,
            step=step,
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            weight_decay=0.01,
        )
        sync_tpu()

    torch.testing.assert_close(p_tpu.cpu(), p_ref, atol=1e-5, rtol=1e-4)


def test_adamw_xla_foreach_step_on_tpu():
    """Verify adamw_xla_foreach_step_ matches single-tensor execution on TPU."""
    torch.manual_seed(101)
    dev = get_tpu_device()
    shapes = [(16, 32), (64,), (32, 32)]

    p_single = [torch.randn(s, device=dev, dtype=torch.float32) for s in shapes]
    p_multi = [p.clone() for p in p_single]
    grads = [torch.randn(s, device=dev, dtype=torch.float32) for s in shapes]

    ea_s = [torch.zeros_like(p) for p in p_single]
    eas_s = [torch.zeros_like(p) for p in p_single]
    ea_m = [torch.zeros_like(p) for p in p_single]
    eas_m = [torch.zeros_like(p) for p in p_single]

    for p, g, ea, eas in zip(p_single, grads, ea_s, eas_s):
        adamw_xla_step_(p, g, ea, eas, step=1, learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01)

    adamw_xla_foreach_step_(
        p_multi, grads, ea_m, eas_m, step=1, learning_rate=1e-3, beta1=0.9, beta2=0.95, eps=1e-8, weight_decay=0.01
    )
    sync_tpu()

    for ps, pm in zip(p_single, p_multi):
        torch.testing.assert_close(pm.cpu(), ps.cpu(), atol=1e-5, rtol=1e-4)


def test_muon_xla_foreach_step_on_tpu():
    """Verify muon_xla_foreach_step_ matches single-tensor execution on TPU."""
    torch.manual_seed(202)
    dev = get_tpu_device()
    shapes = [(32, 32), (64, 32)]

    p_single = [torch.randn(s, device=dev, dtype=torch.bfloat16) for s in shapes]
    p_multi = [p.clone() for p in p_single]
    grads = [torch.randn(s, device=dev, dtype=torch.bfloat16) for s in shapes]
    m_single = [torch.zeros_like(p) for p in p_single]
    m_multi = [torch.zeros_like(p) for p in p_single]

    for p, g, m in zip(p_single, grads, m_single):
        muon_xla_step_(p, g, m, learning_rate=0.02, momentum=0.95, weight_decay=0.01, nesterov=True)

    muon_xla_foreach_step_(
        p_multi, grads, m_multi, learning_rate=0.02, momentum=0.95, weight_decay=0.01, nesterov=True
    )
    sync_tpu()

    for ps, pm in zip(p_single, p_multi):
        torch.testing.assert_close(pm.cpu(), ps.cpu(), atol=2e-3, rtol=2e-3)


def test_adamw_baseline_optimizer_step_on_tpu():
    """Verify AdamW baseline optimizer executes full training step on TPU."""
    from cauchylift.baselines.adamw import AdamW
    torch.manual_seed(42)
    dev = get_tpu_device()

    model = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.Linear(64, 16),
    ).to(device=dev, dtype=torch.bfloat16)

    opt = AdamW(model.parameters(), lr=1e-3, backend="auto")
    x = torch.randn(4, 32, device=dev, dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()

    opt.step()
    opt.zero_grad()
    sync_tpu()

    for p in model.parameters():
        assert torch.isfinite(p).all(), "Non-finite parameter after AdamW on TPU"


def test_cauchylift_theorems_on_tpu():
    """Verify CauchyLift paper theorems (Scale Invariance, Coordinate Bounds, Strict Descent) on TPU."""
    import math
    torch.manual_seed(42)
    dev = get_tpu_device()

    # Theorem 1: Scale Invariance
    M = torch.randn(32, 64, device=dev)
    u_base = cauchylift_direction_xla(M)
    sync_tpu()
    for alpha in [0.01, 2.0, 100.0]:
        u_scaled = cauchylift_direction_xla(alpha * M)
        sync_tpu()
        torch.testing.assert_close(u_scaled.cpu(), u_base.cpu(), atol=1e-5, rtol=1e-4)

    # Theorem 2: Coordinate Bounds
    shape = (32, 64)
    m, n = shape
    bound = min(math.sqrt(m), math.sqrt(n))
    M = torch.randn(shape, device=dev)
    squares = M.square()
    row_rms = (squares.sum(dim=1, keepdim=True) / n).sqrt()
    col_rms = (squares.sum(dim=0, keepdim=True) / m).sqrt()
    Z = M / (row_rms + col_rms).clamp_min(1e-12)
    sync_tpu()
    assert float(Z.abs().max().item()) <= bound + 1e-5

    # Theorem 3: Strict Descent Alignment
    u = cauchylift_direction_xla(M)
    sync_tpu()
    dot = float((M * u).sum().item())
    assert dot > 0.0, f"Inner product on TPU must be positive: {dot}"


