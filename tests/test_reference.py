"""Unit and regression tests for the CauchyLift optimizer and reference implementations."""

from __future__ import annotations

import math
import pytest
import torch

from cauchylift import CauchyLift, cauchylift_direction, cauchylift_reference_step


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


def test_theorem1_degree0_scale_invariance():
    """Theorem 1: U(alpha * M) = U(M) for any alpha > 0."""
    torch.manual_seed(42)
    M = torch.randn(32, 64)
    u_base = cauchylift_direction(M)

    for alpha in [1e-3, 0.1, 2.0, 50.0, 1000.0]:
        u_scaled = cauchylift_direction(alpha * M)
        torch.testing.assert_close(u_scaled, u_base, atol=1e-6, rtol=1e-5)


def test_theorem2_coordinate_magnitude_bounds():
    """Theorem 2: |Z_ij(M)| <= min(sqrt(n), sqrt(m))."""
    torch.manual_seed(1337)
    for shape in [(16, 64), (64, 16), (32, 32), (1, 100), (100, 1)]:
        M = torch.randn(shape)
        m, n = shape
        bound = min(math.sqrt(m), math.sqrt(n))

        squares = M.square()
        row_rms = (squares.sum(dim=1, keepdim=True) / n).sqrt()
        col_rms = (squares.sum(dim=0, keepdim=True) / m).sqrt()
        denom = row_rms + col_rms
        Z = M / denom

        max_val = float(Z.abs().max().item())
        assert max_val <= bound + 1e-6, f"Coordinate bound violated: {max_val} > {bound} for shape {shape}"


def test_theorem3_strict_descent_alignment():
    """Theorem 3: <M, U(M)>_F > 0 for any non-zero matrix M."""
    torch.manual_seed(999)
    for shape in [(16, 64), (64, 16), (32, 32), (1, 100), (100, 1)]:
        M = torch.randn(shape)
        u = cauchylift_direction(M)
        dot_product = float((M * u).sum().item())
        assert dot_product > 0.0, f"Inner product not strictly positive: {dot_product} for shape {shape}"

