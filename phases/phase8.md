# Phase 8 Prompt — Flagship 350M-Parameter, 3B-Token Experiment on 8x TPU v6e

Work autonomously in the CauchyLift repository and complete Phase 8. Read `phases/README.md` and require a PASS handoff from Phase 7. This phase executes the frozen 350M flagship confirmatory protocol; it does not alter optimizer designs or tuning grids.

## Objective

Train the frozen **350M-parameter** decoder-only Transformer for exactly **3,000,000,000** non-padding FineWeb-Edu training tokens per run on the **8x Google Cloud TPU v6e (Trillium)** cluster, comparing CauchyLift against tuned AdamW and Muon across three confirmatory seeds (`[42, 43, 44]`). Measure convergence speed, final perplexity, zero-shot benchmarks, and memory scaling under Chinchilla-aligned token-to-parameter budgets.

## Required Work

1. **Preflight Verification:**
   - Verify git commit, protocol hashes (`protocol_350m_fineweb.json`), FineWeb-Edu shard revision, tokenizer, 350M parameter count ($\pm 2\%$), TPU interconnect health, and HBM margins across all 8 chips.
2. **Flagship Multi-Core Orchestration on 8x TPU v6e:**
   - Launch runs using Torch-XLA PJRT distributed execution with attention and BF16 mixed precision.
   - Effective batch size: 512K tokens (e.g. 256 sequences of length 2048, or 128 sequences of length 4096).
   - Track memory savings: measure persistent HBM allocated across chips (CauchyLift 1 state tensor per parameter vs AdamW 2 state tensors per parameter).
3. **Execution of 3B-Token Flagship Runs:**
   - Execute CauchyLift, AdamW, and Muon across seeds `[42, 43, 44]`.
   - Each run consumes exactly 3,000,000,000 tokens from the identical FineWeb-Edu stream.
   - Strictly frozen hyperparameters: no adjustments to learning rates, warmup, batch sizes, or radius based on intermediate loss curves.
4. **Comprehensive Evaluation & Monitoring:**
   - Log cross-entropy training loss, validation loss on separate held-out FineWeb-Edu partition every 50M tokens, MFU, throughput (tokens/sec), and step time.
   - Run zero-shot evaluations on standard benchmarks (e.g. LAMBADA, PIQA, HellaSwag, WikiText-103) at major checkpoints.
   - Track gradient norm stability, parameter Frobenius norms, and absence of loss spikes throughout the full 3B tokens.
5. **Atomic Checkpoints & Completion Audit:**
   - Save atomic checkpoints every 500M tokens and at run completion.
   - Confirm consumption of exactly 3,000,000,000 non-padding tokens per run.
   - Compute SHA256 hashes of all checkpoints, logs, and evaluation metrics.

## Gate

Phase 8 passes only if:
- All frozen 350M / 3B-token runs across CauchyLift, AdamW, and Muon complete across all seeds or follow registered failure rules;
- Exactly 3B FineWeb-Edu tokens are consumed per run on the 8x TPU v6e cluster;
- CauchyLift maintains competitive or superior convergence and perplexity compared to AdamW while preserving 50% persistent state memory savings;
- No unresolved numerical instability, NaNs, or inter-rank drift occurred;
- Checksums and telemetry artifacts are verified and stored.

Write the standard Phase 8 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase8.json`). Commit without force. Do not start Phase 9 in this session.
