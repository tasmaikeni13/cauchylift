# Phase 7 Prompt — Confirmatory 125M-Parameter, 3B-Token Experiment on 8x TPU v6e

Work autonomously in the CauchyLift repository and complete Phase 7. Read `phases/README.md` and require a PASS handoff from Phase 6. This phase executes the frozen 125M confirmatory protocol; it does not alter optimizer designs or tuning grids.

## Objective

Train the frozen **125M-parameter** decoder-only Transformer for exactly **3,000,000,000** non-padding FineWeb-Edu training tokens per run across an **8x Google Cloud TPU v6e (Trillium)** cluster, for CauchyLift and frozen baselines (AdamW, Muon) across three confirmatory seeds (`[42, 43, 44]`). Produce complete, resumable, auditable empirical evidence.

## Required Work

1. **Preflight Verification:**
   - Verify git commit, protocol hashes (`protocol_125m_fineweb.json`), FineWeb-Edu shard revision, tokenizer, 125M parameter count ($\pm 2\%$), and TPU health across all 8 chips.
   - Refuse execution if protocol drift is detected. Record preflight diagnostics in `artifacts/phase7/preflight.json`.
2. **Distributed Execution:**
   - Launch runs using Torch-XLA PJRT distributed execution with attention and BF16 mixed precision.
   - Present identical token streams for each seed across all compared optimizers.
   - Strictly adhere to preregistered learning rates, warmup, batch sizes, and cosine decay schedules.
3. **Telemetry & Monitoring:**
   - Log cross-entropy training loss, validation loss on separate held-out FineWeb-Edu partition every 50M tokens, MFU, token throughput, and step time.
   - Maintain compact structured telemetry (`metrics.jsonl`) with $< 1\%$ monitoring overhead.
4. **Fault Tolerance & Atomic Checkpoints:**
   - Save atomic checkpoints every 250M tokens and at run completion.
   - Verify deterministic resumption from interrupted cursors in case of transient node interruption.
5. **Completion Audit:**
   - Confirm consumption of exactly 3,000,000,000 non-padding tokens per run.
   - Confirm zero data leakage between training and validation sets.
   - Compute SHA256 hashes of all checkpoints, logs, and telemetry.

## Gate

Phase 7 passes only if:
- All frozen optimizer runs complete across seeds `[42, 43, 44]` or follow registered failure rules;
- Exactly 3B FineWeb-Edu tokens are consumed per run on the 8x TPU v6e cluster;
- CauchyLift demonstrates competitive or superior validation loss and perplexity compared to AdamW while preserving 50% optimizer state memory savings;
- No unresolved NaNs, gradient explosions, or inter-rank drift occurred;
- Checksums and telemetry artifacts are verified and stored.

Write the standard Phase 7 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase7.json`). Commit without force. Do not start Phase 8 in this session.
