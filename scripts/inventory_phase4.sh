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
lspci -nn | grep -Ei 'amd|vga|display|3d'

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
python3 -m pip --version 2>&1 || true

section "PINNED PROJECT PYTHON, ROCM, AND FLASH ATTENTION SMOKE"
.venv/bin/python - <<'PY'
import json
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

record = {
    "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    "device_count": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
    "hip_runtime": torch.version.hip,
    "torch": torch.__version__,
}
print(json.dumps(record, indent=2, sort_keys=True))

if torch.cuda.is_available():
    q = torch.randn(2, 4, 64, 32, dtype=torch.bfloat16, device="cuda")
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        out = torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
        torch.cuda.synchronize()
        print("flash_attention_smoke_shape", list(out.shape), "norm", float(out.norm()))
PY

section "TOKENIZER SMOKE"
.venv/bin/python - <<'PY'
import tiktoken
enc = tiktoken.get_encoding("gpt2")
sample = enc.encode("CauchyLift Phase 4 reproduction test.")
print("tokenizer_name: gpt2", "vocab_size:", enc.n_vocab, "encoded_tokens:", sample)
PY

section "PROJECT LOCK"
.venv/bin/python -m pip freeze --all

section "HOST MEMORY AND DISK"
free -h
df -h /root/cauchylift /tmp
ulimit -a

section "PROFILERS"
for tool in rocprof rocprofv2 rocprofv3 omniperf rocscope rocprof-compute rocprof-sys; do
  command -v "$tool" || true
done
rocprofv3 --version 2>&1 || true

section "GIT"
git status --short --branch
git log -3 --oneline
