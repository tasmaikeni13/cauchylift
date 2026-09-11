import pytest
import torch
from cauchylift.xla import is_tpu_available

# Direction-space tolerances after quantization of the represented input gradient.
ORACLE_FP64 = {"rtol": 5e-12, "atol": 5e-12}
REFERENCE_FP32 = {"rtol": 5e-5, "atol": 2e-5}
TPU_FP32 = {"rtol": 4e-4, "atol": 2e-4}
TPU_BF16 = {"rtol": 4e-3, "atol": 2e-3}
BF16_UPDATE = {"rtol": 2e-2, "atol": 2e-2}


def pytest_collection_modifyitems(config, items):
    tpu_avail = is_tpu_available()
    skip_tpu = pytest.mark.skip(reason="requires an available Google Cloud TPU device")
    for item in items:
        if "tpu" in item.keywords and not tpu_avail:
            item.add_marker(skip_tpu)
        if "rocm" in item.keywords:
            item.add_marker(pytest.mark.skip(reason="ROCm not supported on TPU"))
