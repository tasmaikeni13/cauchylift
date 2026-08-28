import pytest
import torch


# Frozen before the first Phase 3 comparison run. Values are direction-space
# tolerances after quantization of the represented input gradient.
ORACLE_FP64 = {"rtol": 5e-12, "atol": 5e-12}
REFERENCE_FP32 = {"rtol": 5e-5, "atol": 2e-5}
HIP_FP32 = {"rtol": 4e-4, "atol": 2e-4}
HIP_BF16 = {"rtol": 4e-3, "atol": 2e-3}
BF16_UPDATE = {"rtol": 2e-2, "atol": 2e-2}


def pytest_collection_modifyitems(config, items):
    rocm_available = bool(torch.cuda.is_available() and torch.version.hip)
    if rocm_available:
        return
    skip = pytest.mark.skip(reason="requires an available PyTorch ROCm device")
    for item in items:
        if "rocm" in item.keywords:
            item.add_marker(skip)
