from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import threading
from typing import Any

import torch

from .reference import cauchylift_reference

_EXTENSION: Any = None
_LOCK = threading.Lock()


def is_rocm_available() -> bool:
    return bool(torch.cuda.is_available() and torch.version.hip)


def load_extension(*, verbose: bool = False) -> Any:
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not is_rocm_available():
        raise RuntimeError("native CauchyLift requires an available PyTorch ROCm device")
    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        import torch.utils.cpp_extension as cpp_extension

        root = pathlib.Path(__file__).resolve().parents[1]
        build_dir = pathlib.Path(
            os.environ.get("CAUCHYLIFT_BUILD_DIR", "/tmp/cauchylift-hip-build")
        )
        build_dir.mkdir(parents=True, exist_ok=True)
        source_dir = build_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        kernel_source = source_dir / "cauchylift_kernel.cu"
        shutil.copy2(root / "csrc" / "cauchylift_kernel.cu", kernel_source)
        scripts_dir = str(pathlib.Path(sys.executable).parent)
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
        sdk_root = subprocess.check_output(
            [str(pathlib.Path(scripts_dir) / "rocm-sdk"), "path", "--root"],
            text=True,
        ).strip()
        cpp_extension.ROCM_HOME = sdk_root
        cpp_extension.HIP_HOME = sdk_root
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
        cpp_extension.load(
            name="cauchylift_hip_v03",
            sources=[str(kernel_source)],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            build_directory=str(build_dir),
            with_cuda=True,
            is_python_module=False,
            verbose=verbose,
        )
        _EXTENSION = torch.ops.cauchylift_native
    return _EXTENSION


def _validate_native_input(gradient: torch.Tensor) -> torch.Tensor:
    if not gradient.is_cuda or not torch.version.hip:
        raise ValueError("native HIP path requires a ROCm tensor")
    if gradient.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("native HIP path supports FP32 and BF16 gradients")
    return gradient.contiguous()


def cauchylift_hip(
    gradient: torch.Tensor, *, strict: bool = True
) -> torch.Tensor:
    grad = _validate_native_input(gradient)
    extension = load_extension()
    direction, status = extension.direction(grad)
    if strict:
        values = status.cpu().tolist()
        if values[2]:
            raise ValueError("CauchyLift rejects nonfinite gradients")
        if values[3]:
            return cauchylift_reference(gradient, accumulation_dtype=torch.float64)
    return direction.reshape(gradient.shape)


@torch.no_grad()
def cauchylift_hip_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    learning_rate: float,
    *,
    strict: bool = True,
) -> torch.Tensor:
    if not parameter.is_contiguous() or parameter.dtype not in (
        torch.float32,
        torch.bfloat16,
    ):
        direction = cauchylift_reference(gradient)
        parameter.add_(direction.to(parameter.dtype), alpha=-learning_rate)
        return torch.tensor([0, 0, 0, 0], dtype=torch.int32)
    grad = _validate_native_input(gradient)
    if grad.dtype != parameter.dtype:
        grad = grad.to(parameter.dtype)
    extension = load_extension()
    status = extension.step_(parameter, grad, float(learning_rate))
    if strict:
        values = status.cpu().tolist()
        if values[2]:
            raise ValueError("CauchyLift rejects nonfinite gradients")
        if values[3]:
            direction = cauchylift_reference(gradient, accumulation_dtype=torch.float64)
            parameter.add_(direction.to(parameter.dtype), alpha=-learning_rate)
    return status


@torch.no_grad()
def cauchylift_hip_foreach_step_(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    learning_rate: float,
    *,
    strict: bool = True,
) -> torch.Tensor:
    """Update a same-dtype tensor list with one native multi-tensor pipeline.

    ``strict=False`` is a caller assertion that every gradient is finite, has
    at least two active entries, and has representable nonzero FP32 squares.
    It skips boundary/status analysis and is intended for prevalidated timing
    or training pipelines. The default retains the complete frozen semantics.
    """
    if not parameters or len(parameters) != len(gradients):
        raise ValueError("parameters and gradients must be nonempty equal-length lists")
    prepared: list[torch.Tensor] = []
    for parameter, gradient in zip(parameters, gradients, strict=True):
        if not parameter.is_contiguous():
            raise ValueError("foreach HIP parameters must be contiguous")
        grad = _validate_native_input(gradient)
        if parameter.dtype != grad.dtype:
            grad = grad.to(parameter.dtype)
        prepared.append(grad)
    extension = load_extension()
    status = extension.foreach_step_(
        parameters, prepared, float(learning_rate), not strict
    )
    if strict:
        values = status.cpu().tolist()
        for parameter, gradient, item in zip(
            parameters, gradients, values, strict=True
        ):
            if item[2]:
                raise ValueError("CauchyLift rejects nonfinite gradients")
            if item[3]:
                direction = cauchylift_reference(
                    gradient, accumulation_dtype=torch.float64
                )
                parameter.add_(direction.to(parameter.dtype), alpha=-learning_rate)
    return status
