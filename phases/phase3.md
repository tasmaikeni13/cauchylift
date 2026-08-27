# Phase 3 prompt — reference optimizer and native ROCm/HIP kernels

Work autonomously in the CauchyLift repository and complete Phase 3. Read phases/README.md and require PASS handoffs from Phases 1 and 2. Implement the frozen machine-readable specification from Phase 2 exactly; do not redesign the mathematics to make the kernel easier.

## Objective

Produce a trustworthy PyTorch reference optimizer and a fast native AMD implementation for the single MI300X. Establish numerical equivalence, boundary safety, linear work, state and memory costs, and optimizer-step performance relative to the best supported AdamW implementation.

## Required work

1. Inventory the server before changing anything: GPU identity and availability, ROCm and HIP versions, kernel and driver status, PyTorch build and HIP runtime, compiler toolchain, supported BF16 operations, memory, disk, and relevant profiler availability. Record exact outputs. Do not modify system ROCm, drivers, or the global Python installation.
2. Create a pinned, reproducible project environment. Prefer repository-local dependency metadata and caches outside Git. Verify a minimal PyTorch ROCm tensor operation before building extensions.
3. Implement a slow FP64 oracle directly from the Phase 2 specification, then a clear PyTorch reference implementation with FP32 reductions and BF16/FP32 parameter support. Cover zero gradients, one-sparse projective limits, nearly singular denominators, non-contiguous tensors, rectangular matrices, one-row and one-column shapes, scalars, parameter groups, mixed dtypes, and tied parameters.
4. Expose a standard optimizer interface without adding temporal state or a conventional fallback. Report exactly which tensors and bytes persist between steps. Any training regularization must be a separately controlled objective or protocol choice and must not be described as part of the new primitive.
5. Implement native HIP kernels for the decisive row/column reductions, denominator field, norm reduction, and parameter update. Minimize global-memory passes and kernel-launch overhead while preserving the exact map. Use current official AMD and PyTorch extension guidance for this machine. A Triton or composable-kernel experiment may be retained as an engineering comparison, but it cannot replace the required HIP path.
6. Build layered tests: FP64 oracle versus PyTorch; PyTorch versus HIP; forward update and zero-gradient behavior; exhaustive small shapes; adversarial dynamic range; random model-shaped tensors; BF16 tolerance; deterministic repeatability where supported; memory leaks; repeated steps; checkpoint and reload. Declare tolerances before looking at failures.
7. Benchmark warmed-up optimizer-only and complete parameter-update time against supported fused or foreach AdamW, the unfused CauchyLift reference, and a memory-copy lower bound. Use Transformer-shaped tensors and a matrix-size sweep. Report median, dispersion, launch count, bytes moved, achieved bandwidth, peak transient memory, and persistent optimizer memory. Synchronize correctly and use a profiler to verify that the HIP path actually ran.
8. Add continuous tests that work without a GPU and ROCm-marked tests that skip honestly on other machines. Document build, test, benchmark, and profiler commands.

## Failure handling

Treat numerical disagreement as an implementation bug until the oracle or theory is proved wrong. Treat excessive time caused by launch count, layout, or reductions as an engineering failure and iterate on the kernel. If failures arise from discontinuity, undefined boundary behavior, impossible width semantics, or another mathematical defect, invoke the theory-repair loop and invalidate this phase. Do not hide a mathematical change inside an epsilon or clipping threshold.

## Gate

Phase 3 passes only if:

- CPU/PyTorch/HIP paths agree within predeclared dtype tolerances over all required shapes and adversarial cases;
- the profiler proves native ROCm/HIP execution on the MI300X;
- arithmetic remains linear in parameter count and no persistent optimizer state has appeared;
- no NaN, Inf, illegal access, leak, or silent fallback occurs in stress tests;
- the fused optimizer-only median is within 15 percent of the best supported AdamW path on the predeclared representative tensor suite, or the phase is marked REVISE rather than waived;
- the repository records that the original two-GPU-family criterion still needs independent external replication and is not satisfied by one MI300X.

Write the standard Phase 3 artifacts, including raw benchmark data and environment hashes. Commit and push validated source and small results without force. Keep build products out of Git. Do not build the language model in this session.
