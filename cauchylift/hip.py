"""ROCm/HIP native kernel loader and dispatch wrapper for CauchyLift."""

from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import threading
from typing import Any

import torch

_EXTENSION: Any = None
_LOCK = threading.Lock()


def is_rocm_available() -> bool:
    """Check if ROCm/HIP device execution is available."""
    return bool(torch.cuda.is_available() and torch.version.hip)


def load_cauchylift_extension(*, verbose: bool = False) -> Any:
    """Lazily compile and load the native ROCm/HIP C++ extension for CauchyLift."""
    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION
    if not is_rocm_available():
        raise RuntimeError("Native CauchyLift requires an available PyTorch ROCm device")
    with _LOCK:
        if _EXTENSION is not None:
            return _EXTENSION
        import torch.utils.cpp_extension as cpp_extension

        pkg_root = pathlib.Path(__file__).resolve().parent
        repo_root = pkg_root.parent
        build_dir = pathlib.Path(
            os.environ.get("CAUCHYLIFT_BUILD_DIR", "/tmp/cauchylift-hip-build")
        )
        build_dir.mkdir(parents=True, exist_ok=True)
        source_dir = build_dir / "sources"
        source_dir.mkdir(parents=True, exist_ok=True)
        kernel_source = source_dir / "cauchylift_kernel.cu"

        # Locate kernel file
        src_path = repo_root / "csrc" / "cauchylift_kernel.cu"
        if not src_path.exists():
            src_path = pkg_root / "csrc" / "cauchylift_kernel.cu"
        shutil.copy2(src_path, kernel_source)

        scripts_dir = str(pathlib.Path(sys.executable).parent)
        os.environ["PATH"] = scripts_dir + os.pathsep + os.environ.get("PATH", "")
        try:
            sdk_root = subprocess.check_output(
                [str(pathlib.Path(scripts_dir) / "rocm-sdk"), "path", "--root"],
                text=True,
            ).strip()
            cpp_extension.ROCM_HOME = sdk_root
            cpp_extension.HIP_HOME = sdk_root
        except Exception:
            pass

        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx942")
        cpp_extension.load(
            name="cauchylift_hip_native",
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


def _validate_native_input(tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_cuda or not torch.version.hip:
        raise ValueError("Native HIP path requires a ROCm tensor")
    if tensor.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError("Native HIP path supports FP32 and BF16 tensors")
    return tensor.contiguous()


@torch.no_grad()
def cauchylift_hip_step_(
    parameter: torch.Tensor,
    gradient: torch.Tensor,
    momentum_buffer: torch.Tensor,
    learning_rate: float,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
) -> torch.Tensor:
    """Execute a single fused CauchyLift step on ROCm GPU."""
    p = _validate_native_input(parameter)
    g = _validate_native_input(gradient)
    m = _validate_native_input(momentum_buffer)
    extension = load_cauchylift_extension()
    return extension.step_(
        p, g, m,
        float(learning_rate),
        float(momentum),
        float(weight_decay),
    )


@torch.no_grad()
def cauchylift_hip_foreach_step_(
    parameters: list[torch.Tensor],
    gradients: list[torch.Tensor],
    momentum_buffers: list[torch.Tensor],
    learning_rate: float,
    momentum: float = 0.95,
    weight_decay: float = 0.01,
) -> torch.Tensor:
    """Execute a multi-tensor foreach fused CauchyLift step on ROCm GPU."""
    if not parameters or len(parameters) != len(gradients) or len(parameters) != len(momentum_buffers):
        raise ValueError("Parameters, gradients, and momentum_buffers must be nonempty equal-length lists")
    p_prep = [_validate_native_input(p) for p in parameters]
    g_prep = [_validate_native_input(g) for g in gradients]
    m_prep = [_validate_native_input(m) for m in momentum_buffers]
    extension = load_cauchylift_extension()
    return extension.foreach_step_(
        p_prep, g_prep, m_prep,
        float(learning_rate),
        float(momentum),
        float(weight_decay),
    )
