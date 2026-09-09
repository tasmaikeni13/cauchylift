# Phase 5 Prompt — Small-Scale Multi-Workload Screen and Baseline Verification

Work autonomously in the CauchyLift repository and complete Phase 5. Read `phases/README.md` and require PASS handoffs through Phase 4. This phase evaluates CauchyLift across multiple neural network architectures against standard baselines.

## Objective

Run predeclared, equal-budget small-scale experiments comparing CauchyLift with AdamW, Muon, and standard baselines across diverse neural architectures. Verify convergence speed, loss stability, generalization, and memory efficiency before large-scale runs.

## Required Work

1. **Experimental Protocol:**
   - Commit an immutable small-scale protocol with model configurations, datasets, token budgets, validation cadences, seeds, and learning rate grids.
   - Hold one workload out from hyperparameter selection.
2. **Workload Suite:**
   - Small Decoder-only Language Model.
   - Medium Decoder-only Language Model.
   - Vision Transformer (ViT).
   - Non-square Convolutional / State-Space Model (ConvSSM).
3. **Multi-Optimizer Baseline Comparisons:**
   - Compare CauchyLift against AdamW, Muon, SOAP, and NormalizedGD under identical data order, token accounting, and cosine learning rate schedules.
   - Measure: tokens-to-target loss, final validation perplexity, step latency, peak persistent VRAM, and gradient norm stability.
4. **Analysis & Telemetry:**
   - Verify that CauchyLift preserves 50% persistent state memory savings compared to AdamW across all models.
   - Verify that CauchyLift exhibits zero loss spikes, NaN events, or numeric divergence.
   - Confirm that CauchyLift's native HIP step latency remains within $1.5\times$ of element-wise AdamW and $>10\times$ faster than Muon.

## Gate

Phase 5 passes only if:
- All predeclared runs and comparisons complete and are recorded in structured logs;
- CauchyLift matches or exceeds tuned AdamW in tokens-to-target on at least 3 of 4 workloads;
- Zero NaNs, gradient explosions, or loss spikes occur across all seeds;
- Persistent optimizer memory is confirmed at exactly 1 tensor per parameter.

Write the standard Phase 5 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase5.json`). Commit without force. Do not start Phase 6 in this session.
