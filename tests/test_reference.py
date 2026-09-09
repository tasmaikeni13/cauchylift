"""Unit and regression tests for the CauchyLift optimizer and reference implementations."""

from __future__ import annotations

import math
import pytest
import torch

from cauchylift import CauchyLift, cauchylift_direction, cauchylift_reference_step
from cauchylift.hip import cauchylift_hip_foreach_step_, cauchylift_hip_step_, is_rocm_available


REQUIRED_SHAPES = [(), (1,), (7,), (1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (2, 3, 4)]


@pytest.mark.parametrize("shape", REQUIRED_SHAPES)
def test_cauchylift_direction_radius_and_shape(shape):
    """Test that direction output matches input shape and satisfies Frobenius radius sqrt(max(m, n))."""
    count = math.prod(shape) if shape else 1
    tensor = torch.linspace(-3.0, 4.0, count, dtype=torch.float32).reshape(shape)
    direction = cauchylift_direction(tensor)
    assert direction.shape == tensor.shape
    if tensor.numel() > 1:
        m = 1 if tensor.ndim <= 1 else (tensor.shape[0] if tensor.ndim == 2 else tensor.shape[0])
        n = tensor.numel() // m
        expected_radius = math.sqrt(max(m, n))
        actual_radius = float(torch.linalg.vector_norm(direction).item())
        assert math.isclose(actual_radius, expected_radius, rel_tol=1e-5)


def test_zero_and_single_entry_directions():
    """Test zero and 1-sparse edge cases."""
    zero = torch.zeros(3, 5)
    d_zero = cauchylift_direction(zero)
    assert torch.equal(d_zero, torch.zeros_like(d_zero))

    one_sparse = torch.zeros(3, 5)
    one_sparse[1, 4] = -7.0
    d_sparse = cauchylift_direction(one_sparse)
    assert math.isclose(float(d_sparse[1, 4]), -math.sqrt(5), rel_tol=1e-5)
    assert torch.count_nonzero(d_sparse) == 1


def test_momentum_accumulation():
    """Verify that momentum buffer accurately computes M_t = beta * M_{t-1} + (1 - beta) * G_t."""
    param = torch.nn.Parameter(torch.zeros(4, 4))
    g1 = torch.ones(4, 4)
    g2 = torch.full((4, 4), 2.0)

    opt = CauchyLift([param], lr=0.01, momentum=0.9, weight_decay=0.0, backend="reference")
    param.grad = g1
    opt.step()

    buf = opt.state[param]["momentum_buffer"]
    # After step 1: M_1 = 0.9 * 0 + 0.1 * 1 = 0.1
    torch.testing.assert_close(buf, torch.full((4, 4), 0.1))

    param.grad = g2
    opt.step()
    # After step 2: M_2 = 0.9 * 0.1 + 0.1 * 2 = 0.09 + 0.2 = 0.29
    torch.testing.assert_close(buf, torch.full((4, 4), 0.29))


def test_decoupled_weight_decay():
    """Verify decoupled weight decay: W = W * (1 - lr * lambda) - lr * U."""
    param = torch.nn.Parameter(torch.full((4, 4), 5.0))
    param.grad = torch.zeros(4, 4)  # Zero grad -> U = 0

    lr = 0.1
    wd = 0.05
    opt = CauchyLift([param], lr=lr, momentum=0.9, weight_decay=wd, backend="reference")
    opt.step()

    expected = 5.0 * (1.0 - lr * wd)
    torch.testing.assert_close(param, torch.full((4, 4), expected))


def test_optimizer_state_and_checkpoint_resumption():
    """Verify state tracking and exact state_dict reload."""
    p1 = torch.nn.Parameter(torch.ones(2, 3))
    p2 = torch.nn.Parameter(torch.ones(4))
    p1.grad = torch.randn(2, 3)
    p2.grad = torch.randn(4)

    opt1 = CauchyLift([p1, p2], lr=0.01, backend="reference")
    opt1.step()

    summary = opt1.persistent_tensor_summary()
    assert summary["tensors"] == 2
    assert summary["bytes"] == (2 * 3 + 4) * 4  # FP32: 10 elements * 4 bytes = 40 bytes

    checkpoint = opt1.state_dict()

    opt2 = CauchyLift([p1, p2], lr=0.01, backend="reference")
    opt2.load_state_dict(checkpoint)

    torch.testing.assert_close(
        opt1.state[p1]["momentum_buffer"], opt2.state[p1]["momentum_buffer"]
    )
    torch.testing.assert_close(
        opt1.state[p2]["momentum_buffer"], opt2.state[p2]["momentum_buffer"]
    )


@pytest.mark.skipif(not is_rocm_available(), reason="ROCm not available")
def test_native_hip_vs_reference_fp32():
    """Verify bitwise-close equivalence between native HIP kernel and PyTorch reference in FP32."""
    torch.manual_seed(42)
    p_ref = torch.randn(64, 128, device="cuda", dtype=torch.float32)
    p_hip = p_ref.clone()
    g = torch.randn(64, 128, device="cuda", dtype=torch.float32)

    m_ref = torch.zeros_like(p_ref)
    m_hip = torch.zeros_like(p_hip)

    lr = 1e-3
    momentum = 0.95
    wd = 0.01

    cauchylift_reference_step(p_ref, g, m_ref, lr=lr, momentum=momentum, weight_decay=wd)
    cauchylift_hip_step_(p_hip, g, m_hip, lr, momentum, wd)

    torch.testing.assert_close(p_hip, p_ref, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(m_hip, m_ref, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not is_rocm_available(), reason="ROCm not available")
def test_native_hip_foreach_equivalence():
    """Verify multi-tensor foreach HIP kernel against single-tensor step."""
    torch.manual_seed(42)
    shapes = [(16, 32), (64, 64), (128, 32)]
    p_single = [torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in shapes]
    p_multi = [p.clone() for p in p_single]
    grads = [torch.randn(s, device="cuda", dtype=torch.bfloat16) for s in shapes]
    m_single = [torch.zeros_like(p) for p in p_single]
    m_multi = [torch.zeros_like(p) for p in p_single]

    lr = 1e-3
    momentum = 0.95
    wd = 0.01

    for p, g, m in zip(p_single, grads, m_single):
        cauchylift_hip_step_(p, g, m, lr, momentum, wd)

    cauchylift_hip_foreach_step_(p_multi, grads, m_multi, lr, momentum, wd)

    for ps, pm in zip(p_single, p_multi):
        torch.testing.assert_close(pm, ps, rtol=1e-3, atol=1e-3)
