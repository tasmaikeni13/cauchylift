# Phase 4 implementation and reproduction

Phase 4 implements a neutral training system for the AMD Instinct MI300X VF (`gfx942`):
- Compact decoder-only Transformer obeying Phase 2 bias-free parameter matrixization
- Verified FlashAttention on ROCm with eager FP32 numerical reference parity
- Baseline optimizer suite (AdamW, Muon, SOAP, SinkGD, NormalizedGD, SignDescent)
- FineWeb-Edu deterministic token streaming, document packing, and partition management
- Atomic resumable checkpointing with exact bitwise deterministic resumption
- Structured local JSONL metrics logging across all 19 contract metrics

## Environment

The validated machine is one AMD Instinct MI300X VF (`gfx942`) with ROCm 10.0 and PyTorch 2.13.0+rocm10.0.0.
Install dependencies into the local `.venv`:

```bash
PIP_CACHE_DIR=/tmp/cauchylift-pip-cache \
  .venv/bin/python -m pip install -r requirements/phase4-requirements.txt
```

## Running tests

Run the complete test suite (90 tests spanning Phase 3 and Phase 4):

```bash
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python -m pytest -q
```

## Running the Phase 4 smoke verification suite

Execute preflight checks, FlashAttention parity, the overfit screen across all 7 optimizers, and checkpoint resumption:

```bash
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python scripts/run_phase4_smoke.py
```

Results are saved to `analysis/results/phase4_smoke_results.json`.

## Capturing profiler traces on MI300X

Profile FlashAttention forward (`attn_fwd`), backward (`bwd_kernel_fuse`), and all custom CauchyLift multi-tensor HIP kernels using `rocprofv3`:

```bash
.venv/bin/rocprofv3 \
  --rocm-root "$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_devel" \
  --disable-signal-handlers true --kernel-trace --stats --summary -f csv json \
  -d artifacts/phase4/profiler -- \
  env PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python benchmarks/profile_phase4.py \
  --optimizer cauchylift --iterations 3
```

## Gate result

**PASS.** All 6 Phase 4 gate requirements are verified on the single MI300X.
