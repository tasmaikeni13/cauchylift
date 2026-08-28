from __future__ import annotations

import itertools
import math

import pytest
import torch

from cauchylift import CauchyLift, cauchylift_oracle, cauchylift_reference
from cauchylift.hip import cauchylift_hip, load_extension
from conftest import BF16_UPDATE, HIP_BF16, HIP_FP32


pytestmark = pytest.mark.rocm


def test_extension_loads_and_device_is_mi300x():
    module = load_extension()
    assert module is not None
    assert torch.version.hip
    assert "MI300X" in torch.cuda.get_device_name(0)


@pytest.mark.parametrize("dtype,tolerance", [(torch.float32, HIP_FP32), (torch.bfloat16, HIP_BF16)])
@pytest.mark.parametrize(
    "shape", [(), (1,), (7,), (1, 1), (1, 17), (17, 1), (2, 3), (3, 2), (17, 5), (2, 3, 4)]
)
def test_hip_required_shapes(dtype, tolerance, shape):
    count = math.prod(shape) if shape else 1
    gradient = torch.linspace(-3.0, 4.0, count, device="cuda").reshape(shape).to(dtype)
    actual = cauchylift_hip(gradient)
    expected = cauchylift_reference(gradient).float()
    torch.testing.assert_close(actual, expected, **tolerance)


@pytest.mark.parametrize("shape", [(1, 1), (1, 3), (2, 2), (2, 3), (3, 2)])
def test_hip_exhaustive_small_shapes(shape):
    for values in itertools.product((-1.0, 0.0, 1.0), repeat=math.prod(shape)):
        gradient = torch.tensor(values, device="cuda").reshape(shape)
        actual = cauchylift_hip(gradient)
        expected = cauchylift_oracle(gradient)
        torch.testing.assert_close(actual.cpu().double(), expected, **HIP_FP32)


def test_hip_boundary_zero_nonfinite_and_rare_path():
    zero = torch.zeros(7, 5, device="cuda")
    assert torch.equal(cauchylift_hip(zero), zero)
    one = zero.clone()
    one[4, 2] = -9
    output = cauchylift_hip(one)
    assert output[4, 2] == -math.sqrt(5)
    assert torch.count_nonzero(output) == 1

    nearly = torch.tensor([[1.0, 1e-30], [0.0, 0.0]], device="cuda")
    actual = cauchylift_hip(nearly, strict=True)
    expected = cauchylift_oracle(nearly)
    torch.testing.assert_close(actual.cpu().double(), expected, **HIP_FP32)

    with pytest.raises(ValueError, match="nonfinite"):
        cauchylift_hip(torch.tensor([1.0, float("nan")], device="cuda"))


@pytest.mark.parametrize("shape", [(257, 64), (64, 64), (256, 128), (32000, 64)])
def test_random_model_shaped_tensors(shape):
    generator = torch.Generator(device="cuda").manual_seed(20260828)
    gradient = torch.randn(shape, generator=generator, device="cuda")
    actual = cauchylift_hip(gradient)
    expected = cauchylift_reference(gradient)
    torch.testing.assert_close(actual, expected, **HIP_FP32)


def test_native_optimizer_forward_fp32_and_bf16():
    for dtype, tolerance in [(torch.float32, HIP_FP32), (torch.bfloat16, BF16_UPDATE)]:
        parameter = torch.nn.Parameter(torch.ones(33, 17, device="cuda", dtype=dtype))
        gradient = torch.linspace(-2, 3, parameter.numel(), device="cuda").reshape_as(parameter).to(dtype)
        parameter.grad = gradient
        expected = parameter.detach().float() - 0.1 * cauchylift_reference(gradient).float()
        optimizer = CauchyLift([parameter], lr=0.1, backend="hip")
        optimizer.step()
        torch.testing.assert_close(parameter.float(), expected, **tolerance)
        assert optimizer.persistent_tensor_summary() == {"tensors": 0, "bytes": 0}


def test_noncontiguous_hip_direction_and_parameter():
    gradient = torch.arange(1, 49, device="cuda", dtype=torch.float32).reshape(6, 8).t()
    assert not gradient.is_contiguous()
    torch.testing.assert_close(
        cauchylift_hip(gradient), cauchylift_reference(gradient), **HIP_FP32
    )

    parameter = torch.nn.Parameter(torch.ones(6, 8, device="cuda").t())
    assert not parameter.is_contiguous()
    parameter.grad = gradient.clone()
    expected = parameter.detach() - 0.1 * cauchylift_reference(gradient)
    CauchyLift([parameter], lr=0.1, backend="hip").step()
    torch.testing.assert_close(parameter, expected, **HIP_FP32)


def test_deterministic_repeatability():
    torch.manual_seed(20260828)
    gradient = torch.randn(513, 257, device="cuda")
    outputs = [cauchylift_hip(gradient) for _ in range(5)]
    torch.cuda.synchronize()
    for output in outputs[1:]:
        # MI300X FP32 atomic reductions do not promise a bitwise summation
        # order. Repeat executions must remain inside the declared HIP bound.
        torch.testing.assert_close(outputs[0], output, **HIP_FP32)


@pytest.mark.stress
def test_repeated_steps_no_nan_inf_or_memory_leak():
    parameter = torch.nn.Parameter(torch.ones(1024, 1024, device="cuda"))
    parameter.grad = torch.randn_like(parameter)
    optimizer = CauchyLift([parameter], lr=1e-4, backend="hip")
    for _ in range(20):
        optimizer.step()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(200):
        optimizer.step()
    torch.cuda.synchronize()
    final = torch.cuda.memory_allocated()
    assert torch.isfinite(parameter).all()
    assert final <= baseline + 4096
    assert optimizer.state == {}


def test_checkpoint_reload_repeated_native_step():
    parameter = torch.nn.Parameter(torch.ones(32, 16, device="cuda"))
    optimizer = CauchyLift([parameter], backend="hip")
    state = optimizer.state_dict()
    clone = CauchyLift([parameter], backend="hip")
    clone.load_state_dict(state)
    for _ in range(4):
        parameter.grad = torch.randn_like(parameter)
        clone.step()
    assert clone.state == {}
    assert torch.isfinite(parameter).all()
