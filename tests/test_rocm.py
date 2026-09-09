"""ROCm/HIP kernel tests on AMD Instinct MI300X."""

from __future__ import annotations

import math
import pytest
import torch

from cauchylift import CauchyLift, cauchylift_reference_step
from cauchylift.hip import (
    cauchylift_hip_foreach_step_,
    cauchylift_hip_step_,
    is_rocm_available,
    load_cauchylift_extension,
)

pytestmark = pytest.mark.skipif(not is_rocm_available(), reason="ROCm not available")


def test_extension_loads_and_device_is_mi300x():
    """Verify ROCm extension loads and targets MI300X."""
    ext = load_cauchylift_extension()
    assert ext is not None
    assert torch.version.hip
    assert "MI300X" in torch.cuda.get_device_name(0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (17, 1), (64, 128), (256, 256)])
def test_hip_step_matches_reference(dtype, shape):
    """Verify single-tensor HIP step matches reference step."""
    torch.manual_seed(42)
    p_ref = torch.randn(shape, device="cuda", dtype=dtype)
    p_hip = p_ref.clone()
    g = torch.randn(shape, device="cuda", dtype=dtype)
    m_ref = torch.zeros_like(p_ref)
    m_hip = torch.zeros_like(p_hip)

    lr = 1e-3
    momentum = 0.95
    weight_decay = 0.01

    cauchylift_reference_step(
        p_ref, g, m_ref, lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    cauchylift_hip_step_(
        p_hip, g, m_hip, learning_rate=lr, momentum=momentum, weight_decay=weight_decay
    )

    atol = 2e-3 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-3 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(p_hip, p_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(m_hip, m_ref, atol=atol, rtol=rtol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_hip_foreach_step_matches_single(dtype):
    """Verify multi-tensor foreach HIP step matches single-tensor execution."""
    torch.manual_seed(1337)
    shapes = [(16, 64), (128, 256), (32, 32)]
    p_single = [torch.randn(s, device="cuda", dtype=dtype) for s in shapes]
    p_multi = [p.clone() for p in p_single]
    grads = [torch.randn(s, device="cuda", dtype=dtype) for s in shapes]
    m_single = [torch.zeros_like(p) for p in p_single]
    m_multi = [torch.zeros_like(p) for p in p_single]

    lr = 2e-3
    momentum = 0.9
    weight_decay = 0.02

    for p, g, m in zip(p_single, grads, m_single):
        cauchylift_hip_step_(p, g, m, learning_rate=lr, momentum=momentum, weight_decay=weight_decay)

    cauchylift_hip_foreach_step_(
        p_multi, grads, m_multi, learning_rate=lr, momentum=momentum, weight_decay=weight_decay
    )

    atol = 2e-3 if dtype == torch.bfloat16 else 1e-5
    rtol = 2e-3 if dtype == torch.bfloat16 else 1e-4
    for ps, pm in zip(p_single, p_multi):
        torch.testing.assert_close(pm, ps, atol=atol, rtol=rtol)


def test_optimizer_full_step_on_rocm():
    """Verify full CauchyLift optimizer step in backend='auto' on MI300X."""
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 128, bias=False),
        torch.nn.Linear(128, 32, bias=False),
    ).cuda().bfloat16()

    opt = CauchyLift(model.parameters(), lr=1e-3, momentum=0.95, weight_decay=0.01, backend="auto")
    x = torch.randn(8, 64, device="cuda", dtype=torch.bfloat16)
    loss = model(x).sum()
    loss.backward()

    opt.step()
    opt.zero_grad()

    summary = opt.persistent_tensor_summary()
    assert summary["tensors"] == 2
    assert summary["bytes"] > 0
