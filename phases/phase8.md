# Phase 8 prompt — frozen 350M-parameter, 3B-token flagship experiment on 8x MI300X

Work autonomously in the CauchyLift repository and complete Phase 8. Read phases/README.md and require a PASS handoff from Phase 7. This phase executes the frozen 350M flagship confirmatory protocol; it does not alter optimizer designs or tuning grids.

## Objective

Train the frozen approximately **350M-parameter** decoder-only Transformer for exactly **3,000,000,000** non-padding FineWeb-Edu training tokens per run on the **8x AMD Instinct MI300X** cluster, comparing CauchyLift v0.3 against tuned AdamW and Muon across three confirmatory seeds (`[42, 43, 44]`). Measure convergence speed, final perplexity, zero-shot benchmarks, and memory efficiency under Chinchilla-aligned token-to-parameter scaling.

## Required work

1. **Preflight Verification:**
   - Verify git commit, protocol hashes (`phase8_350m_protocol.json`), dataset revision, tokenizer, 350M parameter count ($\pm 2\%$), environment, RCCL bandwidth across the 8 MI300Xs, disk space, and VRAM margins.
   - Refuse to execute if protocol drift is detected. Record preflight diagnostics in `artifacts/phase8/preflight.json`.
2. **Flagship Multi-GPU Orchestration on 8x MI300X:**
   - Launch runs using `torchrun --nproc_per_node=8` with FlashAttention and BF16 mixed precision.
   - Global batch size: e.g. 512K tokens (256 sequences of length 2048).
   - Ensure DDP all-reduce operates with maximum communication-computation overlap.
   - Track memory savings: measure persistent VRAM allocated across ranks (CauchyLift 0 GB state memory vs AdamW ~2.8 GB state memory per rank).
3. **Execution of Frozen 3B-Token Flagship Runs:**
   - Execute CauchyLift v0.3, AdamW, and Muon across seeds `[42, 43, 44]`.
   - Each run consumes exactly 3,000,000,000 tokens from the identical FineWeb-Edu stream for that seed.
   - Strictly frozen hyperparameters: no adjustments to learning rates, warmup, batch size, or radius based on intermediate loss curves.
4. **Comprehensive Evaluation & Monitoring:**
   - Log training cross-entropy loss, validation loss on separate held-out FineWeb-Edu partition every 50M tokens, MFU, token throughput, and step time.
   - Run zero-shot / validation evaluations on standard benchmarks (e.g. LAMBADA, PIQA, HellaSwag, or WikiText-103 perplexity) at checkpoints.
   - Verify that CauchyLift preserves gradient-update alignment, stable rank, and absence of loss spikes throughout the full 3B tokens.
5. **Atomic Checkpointing & Fault Tolerance:**
   - Save atomic checkpoints every 500M tokens and at run completion.
   - Guarantee exact reproducibility and seamless resumption from any interrupted token step.
6. **Completion Verification:**
   - Confirm consumption of exactly 3,000,000,000 non-padding tokens per run.
   - Ensure zero data contamination. Compute SHA256 hashes of all checkpoints, logs, and summary metrics.

## Gate

Phase 8 passes only if:

- all frozen 350M / 3B-token runs across CauchyLift v0.3, AdamW, and Muon complete across all seeds or are accounted for under preregistered crash rules;
- exactly 3B FineWeb-Edu tokens are consumed per run on the 8x MI300X cluster;
- CauchyLift v0.3 maintains competitive or superior convergence and perplexity compared to AdamW while maintaining 0 persistent state;
- no unresolved numerical instability, NaN, loss spike, or inter-rank drift occurred;
- all checkpoints, raw step telemetry, and evaluation summaries are verified and hashed.

Write the standard Phase 8 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase8.json`). Commit and push without force. Do not launch Phase 9 in this session.
