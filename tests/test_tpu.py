"""Google Cloud TPU / XLA tests for CauchyLift."""

from __future__ import annotations

import pytest
import torch

from cauchylift import CauchyLift, cauchylift_reference_step
from cauchylift.xla import (
    cauchylift_direction_xla,
    cauchylift_xla_foreach_step_,
    cauchylift_xla_step_,
    get_hlo_text,
    get_tpu_device,
    is_tpu_available,
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
