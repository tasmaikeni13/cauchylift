#!/usr/bin/env bash
set +e

section() {
  printf '\n===== %s =====\n' "$1"
}

section "UTC"
date --utc --iso-8601=seconds

section "OS AND KERNEL"
uname -a
cat /etc/os-release

section "GPU PCI"
lspci -nn | rg -i 'amd|vga|display|3d'

section "ROCM INFO"
rocminfo

section "ROCM SMI"
rocm-smi --showproductname --showuniqueid --showdriverversion --showvbios \
  --showmeminfo vram --showuse --showtemp

section "HIP CONFIG"
hipconfig --full

section "COMPILERS"
hipcc --version
amdclang++ --version
cmake --version
.venv/bin/ninja --version

section "GLOBAL PYTHON"
python3 --version
python3 -m pip --version
python3 - <<'PY'
try:
    import torch
    print("global_torch", torch.__version__, torch.version.hip)
except Exception as error:
    print(type(error).__name__ + ": " + str(error))
PY

section "PINNED PROJECT PYTHON AND ROCM SMOKE"
.venv/bin/python - <<'PY'
import json
import torch

record = {
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "hip_runtime": torch.version.hip,
    "torch": torch.__version__,
}
print(json.dumps(record, indent=2, sort_keys=True))
if torch.cuda.is_available():
    value = (torch.ones(8, device="cuda", dtype=torch.bfloat16) ** 2).sum()
    torch.cuda.synchronize()
    print("bf16_smoke_sum", float(value))
PY

section "PROJECT LOCK"
.venv/bin/python -m pip freeze --all
sha256sum requirements/rocm10-mi300x.txt requirements/phase3-lock.txt

section "HOST MEMORY AND DISK"
free -h
df -h /root/cauchylift /tmp
ulimit -a

section "PROFILERS"
for tool in rocprof rocprofv2 rocprofv3 omniperf rocscope rocprof-compute rocprof-sys; do
  command -v "$tool" || true
done
rocprofv3 --version

section "GIT"
git status --short --branch
git log -3 --oneline
