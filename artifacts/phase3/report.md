# Phase 3 reference optimizer and native ROCm/HIP kernels

Run ID: `phase3-20260828T180051Z`

Audit date: 2026-08-28 UTC

Required handoffs: Phase 1 `PASS` at `5221b458405119b0da297500b210ddc7a34bf0c5`; Phase 2 `PASS` at `ef6c726b62cf693216aedcf42afe8669c0a2719e`; both verified on `origin/main` before implementation

Scope: frozen optimizer implementation, correctness/stress tests, MI300X kernel benchmark and profiling; no model, dataset, or training loop

## Decision summary

**REVISE.** The FP64 oracle, PyTorch reference, standard state-free optimizer,
and native HIP implementation satisfy the numerical, boundary, safety,
complexity, persistent-state, memory-leak, and native-execution requirements.
All 63 tests pass. ROCprofiler proves execution of the custom marginal,
exclusion, denominator, norm, and parameter-update kernels on the single
MI300X.

The mandatory performance gate fails and is not waived. On the predeclared
123,586,560-element BF16 Transformer-shaped suite, optimized native CauchyLift
takes a 5.3505 ms optimizer-only median; the best supported AdamW path is fused
AdamW at 0.9076 ms. The ratio is 5.8953, above the required 1.15. Phase 4 is not
authorized.

## Requirement-level evidence

| Requirement | Result | Decisive evidence |
|---|---|---|
| Server inventory before changes | Complete | `environment.txt` records Ubuntu 24.04.4, kernel 6.8.0-138, ROCk 6.19.14.31400000, MI300X VF `gfx942`, HIP runtime 7.15.26333, ROCm 10.0, 205,822,885,888 bytes VRAM, host/disk, compilers, BF16, and rocprofv3 1.3.5. The initial global Python had no pip or torch; system ROCm/driver were not modified. |
| Pinned isolated environment | Complete | `.venv`, `requirements/rocm10-mi300x.txt`, and full resolved lock. The AMD `torch[device-gfx942]==2.13.0+rocm10.0.0` wheel passes a BF16 ROCm tensor smoke operation. Caches/builds remain under `/tmp`. |
| FP64 oracle and PyTorch reference | Complete | Independent CPU FP64 direct-exclusion oracle; prefix/suffix PyTorch reference; FP32 minimum accumulation and explicit FP64 rare path. Tests cover zero, exact one-sparse, near-singular, scalar/vector/matrix/higher tensors, rectangular/one-row/one-column shapes, noncontiguous tensors, sparse materialization, mixed dtypes, and adversarial range. |
| Standard optimizer and state | Complete | `CauchyLift` implements parameter groups, tied-parameter deduplication, sparse-to-equivalent-dense handling, mixed BF16/FP32 parameters, closure semantics, checkpoint/reload, and `state == {}`. Persistent summary is exactly zero tensors and zero bytes. Weight decay and conventional fallback options do not exist. |
| Native HIP | Complete for correctness | Nine named custom kernels implement metadata reduction, tiled row/column energy, exclusion scan, active denominator minimum, norm reduction, and fused parameter update. All work is linear in parameter count; no cancellation-prone `2S-r-c`, epsilon, or fallback optimizer appears. |
| Layered tests | Complete | 27 CPU/reference and 36 ROCm tests pass. The suite includes exhaustive values in all declared small shapes, oracle/reference/HIP agreement, FP32/BF16 updates, random model shapes, nonfinite rejection, repeatability within the predeclared atomic-reduction tolerance, 200 repeated steps, memory stability, and checkpoint/reload. |
| Benchmark | **Failed speed gate** | Raw warmed samples are in both benchmark JSON files. Initial contended native path: 141.0779 ms. Optimized native path: 5.3505 ms. Fused AdamW: 0.9076 ms; foreach AdamW: 2.0176 ms; PyTorch reference: 53.1665 ms; copy lower bound: 0.2598 ms. |
| Continuous tests and documentation | Complete | Pytest's `rocm` marker skips honestly without ROCm. `docs/phase3.md` records environment, build, tests, declared tolerances, benchmark, profiler, state, and replication boundary. |

## Numerical and safety audit

The declared comparison tolerances were committed to `tests/conftest.py` before
the first comparison: FP64 5e-12/5e-12, FP32 reference 5e-5/2e-5, HIP FP32
4e-4/2e-4, HIP BF16 direction 4e-3/2e-3, and BF16 update 2e-2/2e-2
(relative/absolute). All comparisons pass without relaxing a tolerance.

Exactly represented zero maps to exact zero. Exactly one-sparse represented
inputs take the signed projective boundary. A non-one-sparse FP32 input whose
active complement rounds to zero takes the declared FP64 recomputation path.
Nonfinite input is rejected by the strict interface. No stress test produced a
NaN, Inf, illegal access, persistent allocation, or parameter-state leak.

MI300X FP32 atomic marginal accumulation does not provide a bitwise summation
order. Five repeated outputs remain within the predeclared HIP tolerance. This
is reported as numerical repeatability, not mislabeled as bitwise determinism.

## Work, launches, and memory

Every full-tensor kernel pass and every row/column exclusion scan is linear in
the number of entries or fibers. The optimized reduction replaced per-entry
contended atomics with tiled partial row/column reductions and two-stage scalar
reductions. That change improved the representative native median by 26.37×.

The raw 4096×4096 profile contains four invocations of each of the nine named
CauchyLift kernels for four optimizer steps. Two FP32 workspace fills make 11
launches per tensor step. Fused AdamW uses two steady-state launches in the
comparison profile. The largest benchmarked CauchyLift transient allocator
increase is 508,416 bytes because tensors are processed sequentially; the
PyTorch reference peaks at 1,896,769,024 transient bytes. CauchyLift persistent
optimizer memory is zero. AdamW stores 219 tensors and 494,346,532 bytes on the
representative BF16 suite.

Logical byte estimates and achieved rates are recorded in the JSON. They are
operation-graph accounting, not hardware-counter byte measurements; the raw
profiler timings are authoritative for dispatch duration.

## Failure and iteration record

1. The minimal host lacked `ensurepip`, PyTorch, Python headers, and complete
   ROCm development headers. The environment was kept local: pip was bootstrapped
   into `.venv`, the official ROCm 10 `gfx942` wheels were installed there, and
   the extension used Python-header-free dispatcher registration.
2. The first compile identified the environment's ROCm-core/devel path split;
   the loader now resolves the pinned `rocm-sdk` root explicitly.
3. `-ffast-math` made the device nonfinite predicate invalid. It was removed;
   the nonfinite rejection regression passes.
4. The first native benchmark was 150.08× fused AdamW because the metadata,
   denominator, norm, and marginal reductions used contended atomics. Tiled and
   hierarchical reductions cut the ratio to 5.8953, but not to 1.15.
5. The system `rocprofv3` conflicted with pip ROCm libraries and aborted. Using
   `.venv/bin/rocprofv3` with the matching pinned devel root produced the
   checked-in raw traces. No driver or global profiler was changed.

Failures are classified as infrastructure (1, 2, 5), implementation (3), and
engineering performance (4). None is a mathematical-specification failure, so
the Phase 1/2 theory-repair loop was not triggered.

## Gate audit

- **PASS:** CPU oracle, PyTorch, and HIP paths agree within every predeclared
  dtype tolerance across required and adversarial cases.
- **PASS:** ROCprofiler proves native HIP execution on the inventoried MI300X.
- **PASS:** arithmetic remains linear and persistent optimizer state is exactly
  zero tensors/zero bytes.
- **PASS:** stress tests show no NaN, Inf, illegal access, leak, or silent
  conventional fallback.
- **REVISE:** 5.8953× fused AdamW is not within 15 percent.
- **OPEN:** the original two-modern-GPU-family criterion still requires
  independent external replication. One MI300X cannot satisfy it.

## Gate result

**REVISE** — the exact Phase 2 map has a trustworthy reference and verified
native MI300X implementation, but its measured optimizer-step performance does
not meet the frozen Phase 3 gate. No language model was built, and no downstream
phase is authorized.

