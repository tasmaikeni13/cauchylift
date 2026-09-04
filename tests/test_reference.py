from __future__ import annotations

import itertools
import math

import pytest
import torch

from cauchylift import CauchyLift, cauchylift_oracle, cauchylift_reference
from conftest import ORACLE_FP64, REFERENCE_FP32


REQUIRED_SHAPES = [(), (1,), (7,), (1, 1), (1, 7), (7, 1), (2, 3), (3, 2), (2, 3, 4)]


@pytest.mark.parametrize("shape", REQUIRED_SHAPES)
def test_oracle_reference_required_shapes(shape):
    count = math.prod(shape) if shape else 1
    gradient = torch.linspace(-3.0, 4.0, count, dtype=torch.float64).reshape(shape)
    actual = cauchylift_reference(gradient, accumulation_dtype=torch.float64)
    expected = cauchylift_oracle(gradient)
    torch.testing.assert_close(actual, expected, **ORACLE_FP64)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (2, 2), (2, 3), (3, 2)])
def test_exhaustive_small_shapes(shape):
    for values in itertools.product((-1.0, 0.0, 1.0), repeat=math.prod(shape)):
        gradient = torch.tensor(values, dtype=torch.float32).reshape(shape)
        expected = cauchylift_oracle(gradient)
        actual = cauchylift_reference(gradient)
        torch.testing.assert_close(actual.double(), expected, **REFERENCE_FP32)


def test_zero_boundary_and_nearly_singular():
    zero, zero_info = cauchylift_reference(
        torch.zeros(3, 5), return_diagnostics=True
    )
    assert torch.equal(zero, torch.zeros_like(zero))
    assert zero_info["zero_gradient_count"] == 1

    one_sparse = torch.zeros(3, 5)
    one_sparse[1, 4] = -7.0
    boundary, boundary_info = cauchylift_reference(
        one_sparse, return_diagnostics=True
    )
    assert boundary[1, 4] == -math.sqrt(5)
    assert torch.count_nonzero(boundary) == 1
    assert boundary_info["one_sparse_boundary_count"] == 1

    nearly = torch.tensor([[1e-25, 1e-25], [0.0, 0.0]], dtype=torch.float32)
    actual, info = cauchylift_reference(nearly, return_diagnostics=True)
    expected = cauchylift_oracle(nearly)
    torch.testing.assert_close(actual.double(), expected, **REFERENCE_FP32)
    assert info["fp64_rare_path_count"] == 1



@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_represented_input_dtypes(dtype):
    gradient = torch.tensor(
        [[1.0, -0.5, 0.0], [0.125, -0.03125, 0.0078125]], dtype=dtype
    )
    actual = cauchylift_reference(gradient)
    expected = cauchylift_oracle(gradient)
    tolerance = ORACLE_FP64 if dtype == torch.float64 else REFERENCE_FP32
    torch.testing.assert_close(actual.double(), expected, **tolerance)


def test_noncontiguous_and_higher_tensor_matrixization():
    base = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)
    gradient = base.t()
    assert not gradient.is_contiguous()
    actual = cauchylift_reference(gradient)
    expected = cauchylift_oracle(gradient)
    torch.testing.assert_close(actual.double(), expected, **REFERENCE_FP32)

    higher = torch.arange(1, 49, dtype=torch.float32).reshape(2, 3, 4, 2)
    output = cauchylift_reference(higher)
    assert output.shape == higher.shape
    assert torch.linalg.vector_norm(output) == pytest.approx(math.sqrt(24), rel=1e-6)


@pytest.mark.parametrize(
    "gradient",
    [torch.tensor([float("nan")]), torch.tensor([float("inf")]), torch.empty(0)],
)
def test_rejects_invalid_inputs(gradient):
    with pytest.raises((ValueError, FloatingPointError)):
        cauchylift_reference(gradient)
    with pytest.raises((ValueError, FloatingPointError)):
        cauchylift_oracle(gradient)


def test_scale_dynamic_range_and_sign():
    gradient = torch.tensor(
        [[1e100, -1e70, 1e40], [1e10, -1e-20, 1e-50]], dtype=torch.float64
    )
    actual = cauchylift_reference(gradient, accumulation_dtype=torch.float64)
    expected = cauchylift_oracle(gradient)
    torch.testing.assert_close(actual, expected, **ORACLE_FP64)
    active = gradient != 0
    assert torch.equal(actual[active].sign(), gradient[active].sign())
    assert torch.isfinite(actual).all()


def test_optimizer_groups_mixed_dtype_tied_state_and_reload():
    p32 = torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32))
    pbf = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    p32.grad = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    pbf.grad = torch.arange(1, 5, dtype=torch.bfloat16)
    optimizer = CauchyLift(
        [
            {"params": [p32, p32], "lr": 0.1},
            {"params": [pbf], "lr": 0.2},
        ],
        backend="reference",
    )
    expected32 = p32.detach() - 0.1 * cauchylift_reference(p32.grad)
    expectedbf = pbf.detach() - 0.2 * cauchylift_reference(pbf.grad).to(torch.bfloat16)
    optimizer.step()
    torch.testing.assert_close(p32, expected32)
    torch.testing.assert_close(pbf, expectedbf)
    assert optimizer.state == {}
    assert optimizer.persistent_tensor_summary() == {"tensors": 0, "bytes": 0}

    checkpoint = optimizer.state_dict()
    clone = CauchyLift(
        [
            {"params": [p32], "lr": 0.1},
            {"params": [pbf], "lr": 0.2},
        ],
        backend="reference",
    )
    clone.load_state_dict(checkpoint)
    assert clone.state == {}
    assert clone.state_dict()["state"] == {}


def test_tied_parameter_conflicting_groups_rejected():
    parameter = torch.nn.Parameter(torch.ones(3))
    with pytest.raises(ValueError, match="conflicting learning rates"):
        CauchyLift(
            [
                {"params": [parameter], "lr": 0.1},
                {"params": [parameter], "lr": 0.2},
            ]
        )


def test_sparse_gradient_is_materialized_without_optimizer_fallback():
    parameter = torch.nn.Parameter(torch.ones(4, 3))
    indices = torch.tensor([[0, 2], [1, 0]])
    values = torch.tensor([2.0, -3.0])
    sparse = torch.sparse_coo_tensor(indices, values, parameter.shape).coalesce()
    parameter.grad = sparse
    expected = parameter.detach() - 0.1 * cauchylift_reference(sparse.to_dense())
    optimizer = CauchyLift([parameter], lr=0.1, backend="reference")
    optimizer.step()
    torch.testing.assert_close(parameter, expected)
