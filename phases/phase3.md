# Phase 3 Prompt — PyTorch Reference and Native ROCm/HIP Fused Multi-Tensor Kernels

Work autonomously in the CauchyLift repository and complete Phase 3. Read `phases/README.md` and require a PASS handoff from Phase 2. This phase develops and benchmarks the high-performance implementation of CauchyLift.

## Objective

Implement, verify, and benchmark the complete CauchyLift optimizer stack:
1. Pure PyTorch reference implementation (`cauchylift/reference.py`) for validation and CPU fallback.
2. Production-grade PyTorch optimizer class (`cauchylift/optimizer.py`).
3. High-performance native fused ROCm/HIP multi-tensor kernel (`csrc/cauchylift_kernel.cu` and `cauchylift/hip.py`) targeting AMD Instinct MI300X (`gfx942`).

## Required Work

1. **Reference Implementation:**
   - Implement `cauchylift_direction` and `cauchylift_reference_step` in `cauchylift/reference.py`.
   - Ensure clean FP32 accumulation, zero-memory leaks, and exact mathematical adherence to Phase 1 & 2 specs.
2. **Native ROCm/HIP Fused Kernel:**
   - Develop tile-parallel HIP kernels fusing:
     - Momentum buffer accumulation: $M_t = \beta M_{t-1} + (1 - \beta) G_t$;
     - Row and column energy reductions via atomic shared/global operations;
     - Additive fiber RMS finalization: $D_{ij} = \text{RMS}_{\text{row}, i} + \text{RMS}_{\text{col}, j}$;
     - Raw norm block reductions: $Z_{ij} = M_{ij}/D_{ij}$, $\|Z\|_F = \sqrt{\sum Z_{ij}^2}$;
     - Parameter update with decoupled weight decay: $W \leftarrow W (1 - \eta \lambda) - \eta \frac{\sqrt{\max(m,n)}}{\|Z\|_F} Z$.
   - Support both single-tensor (`cauchylift_step_hip`) and multi-tensor foreach (`cauchylift_foreach_step_hip`) dispatches in FP32 and BF16.
3. **Verification & Testing:**
   - Test numerical equivalence between native HIP kernels and reference implementation across all standard shapes and dtypes.
   - Verify state dictionary saving and loading (checkpoint resumption).
   - Verify persistent tensor memory: exactly one momentum tensor per parameter (4 bytes/param in FP32).
4. **Kernel Micro-benchmarking:**
   - Benchmark kernel step latency on AMD Instinct MI300X across standard LLM projection dimensions ($768 \times 768$, $2048 \times 768$, $4096 \times 4096$).
   - Confirm step latency $< 0.8$ ms (at least $10\times$ faster than iterative polar/Newton–Schulz methods).

## Gate

Phase 3 passes only if:
- All unit and equivalence tests pass with 100% success on ROCm;
- Native HIP kernel matches reference step within floating-point tolerance (rtol $< 10^{-4}$ for FP32, $< 2\times 10^{-3}$ for BF16);
- Optimizer state tracks exactly 1 persistent tensor per parameter;
- Foreach kernel dispatch executes cleanly across multi-layer models.

Write the standard Phase 3 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase3.json`). Commit without force. Do not start Phase 4 in this session.
