# Phase 3 reference optimizer and native ROCm/HIP kernels

Run ID: `phase3-20260829T042638Z`

Audit date: 2026-08-29 UTC

Required handoffs: Phase 1 `PASS` at
`5221b458405119b0da297500b210ddc7a34bf0c5`; Phase 2 `PASS` at
`ef6c726b62cf693216aedcf42afe8669c0a2719e`; prior Phase 3 `REVISE` at
`2970863`

Scope: complete the frozen Phase 3 implementation and performance gate on the
single inventoried MI300X. No model, dataset, training loop, or Phase 4 work was
performed.

## Decision summary

**PASS.** The FP64 oracle, PyTorch reference, standard state-free optimizer,
and native HIP implementation satisfy the numerical, boundary, safety,
complexity, persistent-state, profiler, and optimizer-step performance gates.
All 68 tests pass.

On the predeclared 123,586,560-element BF16 Transformer-shaped suite, the final
native CauchyLift optimizer-only median is 1.062472 ms (0.004731 ms MAD). Fused
AdamW is the best supported baseline at 0.960179 ms (0.016638 ms MAD). The ratio
is 1.106535, below the mandatory 1.15 maximum.

The original two-modern-GPU-family criterion remains unreplicated. This report
establishes only the required single-MI300X Phase 3 result and does not convert
one machine into external replication.

## Requirement-level evidence

| Requirement | Result | Decisive evidence |
|---|---|---|
| Server inventory | Complete | `environment.txt` records Ubuntu 24.04.4, kernel 6.8.0-138, ROCk 6.19.14.31400000, MI300X VF `gfx942`, HIP runtime 7.15.26333, ROCm 10.0, VRAM, host/disk, compilers, BF16, and rocprofv3 1.3.5. System ROCm, drivers, and global Python were not modified. |
| Pinned environment | Complete | Repository-local `.venv`; direct and resolved requirements under `requirements/`; extension build and caches outside Git. |
| Oracle/reference semantics | Pass | Independent FP64 direct-exclusion oracle and FP32 PyTorch reference cover zero, one-sparse, near-singular, scalar/vector/matrix/higher, noncontiguous, sparse, mixed-dtype, and adversarial cases. |
| Standard optimizer/state | Pass | Parameter groups, tied-parameter deduplication, closure/checkpoint behavior, and supported dtype routing pass. `optimizer.state` is empty: zero persistent tensors and zero bytes. |
| Native HIP | Pass | Same-dtype contiguous ROCm tensors are batched into one metadata table and five custom kernels: parallel initialization, tiled row/column/total energy, raw norm, scale finalization, and fused update. |
| Layered/stress tests | Pass | 27 CPU and 41 ROCm tests pass (68 total), including strict and prevalidated multi-tensor paths, FP32/BF16 agreement, boundary/rare-path behavior, repeatability tolerance, 200 repeated steps, memory stability, and checkpoint/reload. |
| Benchmark | **Pass** | Raw 10-warmup/30-sample data are in `analysis/results/phase3_benchmark.json`. CauchyLift/fused-AdamW is 1.106535, within 1.15. |
| Profiler | Pass | Fresh raw ROCprofiler trace/stats CSV files show all five named custom HIP kernels executing four times on the MI300X. |
| Documentation/CI | Pass | `docs/phase3.md` records environment, build/test/benchmark/profile commands, tolerances, fast-path contract, state, and replication boundary. ROCm tests skip honestly elsewhere. |

## Optimized execution

The optimizer groups eligible contiguous parameters by dtype and performs one
multi-tensor native invocation per group. The frozen denominator is evaluated
as `2S-r_i-c_j`: each 64x64 tile contributes row, column, and total energy in a
single pass. This removes the old per-tensor exclusion, denominator-minimum,
and reduction launch chain. The norm and update passes use different 512-thread
layouts selected for their respective reduction and coalescing behavior.

The workspace initializer is grid-parallel instead of using one block to clear
all row and column buffers. The final device graph has five custom kernel
launches plus one asynchronous metadata-copy dispatch per optimizer step. All
arithmetic is linear in parameter count, and the final representative transient
allocator increase is 876,544 bytes.

`strict=True` retains full status analysis, nonfinite rejection, projective
one-sparse behavior, and the FP64 rare path for rounded invalid denominators.
The timing path uses `strict=False` only after explicitly checking finite input,
at least two active entries, representable active FP32 squares, finite FP32
total energy, and a positive finite denominator lower bound. No epsilon,
clipping, conventional optimizer fallback, or mathematical redesign was used.

## Benchmark details

| Path | Optimizer-only median (ms) | MAD (ms) | Complete-update median (ms) | Persistent bytes |
|---|---:|---:|---:|---:|
| Copy lower bound | 0.264101 | 0.002245 | 0.425306 | 0 |
| Fused AdamW | 0.960179 | 0.016638 | 0.983714 | 494,346,532 |
| Foreach AdamW | 2.086855 | 0.010183 | 2.140355 | 494,346,532 |
| PyTorch CauchyLift reference | 55.433283 | 0.270342 | 56.156067 | 0 |
| Native CauchyLift | 1.062472 | 0.004731 | 1.082177 | 0 |

The declared native logical-operation graph accounts for 4,234,340,064 bytes
on the representative suite. This is semantic load/store/RMW accounting, not
a hardware-counter HBM measurement. Raw event samples and the matrix-size sweep
remain authoritative in the benchmark JSON.

## Profiler evidence

The current profile uses four FP32 optimizer steps on one 4096x4096 tensor. The
raw trace contains four calls to each custom kernel. Mean dispatch durations are
68.796 us for marginal energy, 60.387 us for raw norm, 56.208 us for output,
3.157 us for scale finalization, and 2.596 us for initialization. The profile is
native-execution evidence; the separate BF16 representative benchmark decides
the performance gate.

## Failure and iteration record

1. The original contended implementation measured 141.0779 ms and was retained
   as `analysis/results/phase3_benchmark_initial.json`.
2. The prior tiled per-tensor implementation reduced this to 5.3505 ms but
   remained 5.8953x fused AdamW, so commit `2970863` correctly marked Phase 3
   `REVISE`.
3. Multi-tensor batching exposed a 512-thread row-index bug that duplicated rows
   16-31 and skipped rows 48-63. A reference regression isolated it; corrected
   FP32/BF16 results now pass.
4. A cooperative persistent-kernel experiment and several alternate workgroup
   layouts were slower and were removed from the final source.
5. Direct total-energy accumulation removed exclusion scans. Per-kernel layouts
   and a grid-parallel initializer reduced the official median to 1.062472 ms.
6. One hardened official run measured 1.150318x, slightly over the gate. It was
   treated as a failure; the initializer fix created the final 1.106535x margin.

These were implementation and engineering failures, not mathematical failures;
the Phase 1/2 theory-repair loop was not triggered.

## Gate audit

- **PASS:** oracle, PyTorch, strict HIP, and prevalidated HIP paths agree within
  every predeclared dtype tolerance over required and adversarial cases.
- **PASS:** ROCprofiler proves native custom HIP execution on the MI300X.
- **PASS:** work remains linear and persistent state is zero tensors/zero bytes.
- **PASS:** stress tests show no NaN, Inf, illegal access, leak, or silent
  conventional-optimizer fallback.
- **PASS:** 1.106535x fused AdamW is within the mandatory 1.15 maximum.
- **OPEN/RECORDED:** independent replication on a second modern GPU family is
  still required by the broader research contract.

## Gate result

**PASS.** Phase 3 is complete on the single MI300X. Phase 4 was not started in
this run.
