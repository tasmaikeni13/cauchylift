# Phase 3 implementation and reproduction

Phase 3 implements the frozen `spec/optimizer_v0.2.json`; it does not add an
epsilon, clipping, momentum, moments, weight decay, or another optimizer.
Scalars reshape to 1×1, vectors to length×1, matrices keep their stored shape,
and higher tensors flatten axis 0 against the remaining axes.

## Environment

The validated machine is one AMD Instinct MI300X VF (`gfx942`) with ROCm 10.0.
Create a repository-local environment without modifying global Python or ROCm:

```bash
python3 -m venv --without-pip .venv
curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/cauchylift-get-pip.py
.venv/bin/python /tmp/cauchylift-get-pip.py
PIP_CACHE_DIR=/tmp/cauchylift-pip-cache \
  .venv/bin/python -m pip install -r requirements/rocm10-mi300x.txt
```

The direct requirements and complete resolved environment are pinned under
`requirements/`. Build products default to `/tmp/cauchylift-hip-build` and do
not enter Git. AMD's ROCm 10 package selector supplied the `gfx942` wheel, and
the implementation follows PyTorch's documented Python-header-free custom-op
registration path:

- https://rocm.docs.amd.com/projects/ai-ecosystem/en/latest/frameworks/pytorch/install.html
- https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html
- https://docs.pytorch.org/docs/stable/notes/hip.html
- https://rocm.docs.amd.com/projects/HIP/en/latest/tutorial/reduction.html
- https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/quick-reference/quick_guide.html

## Build and test

The HIP extension builds lazily on first use:

```bash
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python -c 'from cauchylift.hip import load_extension; load_extension(verbose=True)'

PYTHONPATH=. .venv/bin/python -m pytest -q tests/test_reference.py
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python -m pytest -q -m rocm
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python -m pytest -q
```

Non-ROCm machines honestly skip tests marked `rocm`; the complete oracle and
PyTorch reference suite remains available on CPU.

The standard interface is:

```python
from cauchylift import CauchyLift

optimizer = CauchyLift(model.parameters(), lr=1e-3, backend="auto")
```

`optimizer.state` remains empty. Exactly zero tensors and zero bytes persist
between steps. Native transient storage is FP32 row/column energies and
exclusions plus bounded reduction partials; it is allocated per invocation.
The `strict=True` default checks native status on the host and raises on a
nonfinite input or invokes the declared FP64 rare path. Benchmarking uses
`strict=False` only for an explicitly prevalidated finite input suite so that
the host status synchronization is not mixed into kernel timing.

## Declared comparison tolerances

These values were frozen in `tests/conftest.py` before the first comparison:

| Comparison | Relative | Absolute |
|---|---:|---:|
| FP64 oracle vs FP64 PyTorch | 5e-12 | 5e-12 |
| FP64 oracle vs FP32 reference | 5e-5 | 2e-5 |
| FP32 reference vs HIP | 4e-4 | 2e-4 |
| BF16 represented input vs HIP FP32 direction | 4e-3 | 2e-3 |
| BF16 parameter update | 2e-2 | 2e-2 |

FP32 tiled atomic marginal accumulation is not bitwise ordered by the MI300X.
Repeated results are required to remain within the declared HIP tolerance.

## Benchmark and profile

```bash
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python benchmarks/benchmark_phase3.py \
  --warmup 5 --iterations 15 \
  --output analysis/results/phase3_benchmark.json

.venv/bin/rocprofv3 \
  --rocm-root "$PWD/.venv/lib/python3.12/site-packages/_rocm_sdk_devel" \
  --disable-signal-handlers true --kernel-trace --stats --summary -f csv json \
  -d /tmp/cauchylift-profiler-hip -- \
  env PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build \
  .venv/bin/python benchmarks/profile_phase3.py \
  --backend cauchylift_hip --iterations 3
```

The benchmark uses warmed ROCm events and synchronizes every sample. It reports
raw samples, median, MAD, p25/p75, optimizer-only and complete-update times,
logical bytes/bandwidth estimates, allocator peak transient memory, and actual
persistent state. The checked-in profiler CSV files are the raw kernel traces.
They prove that the custom HIP kernels executed; four profiled steps launch each
of the nine named CauchyLift kernels four times. Including two FP32 workspace
fills, the native path uses 11 launches per tensor step versus two steady-state
launches for fused AdamW in this profile.

The optimized native result does not pass the speed gate. The representative
BF16 Transformer-shaped suite takes 5.3505 ms versus 0.9076 ms for fused AdamW,
a ratio of 5.8953 rather than the required maximum 1.15. The initial contended
atomic reduction took 141.0779 ms; that failed run is preserved alongside the
optimized result. Phase 3 is therefore `REVISE`, not `PASS`.

The original research contract requires the timing criterion on two modern GPU
families. This single MI300X result does not satisfy independent external
replication, regardless of its outcome.
