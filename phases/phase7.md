# Phase 7 prompt — frozen 125M-parameter, 1B-token experiment on 8x MI300X

Work autonomously in the CauchyLift repository and complete Phase 7. Read phases/README.md and require a PASS handoff from Phase 6. This phase executes the frozen 125M confirmatory protocol; it does not design or tune the optimizer.

## Objective

Train the frozen approximately **125M-parameter** decoder-only Transformer for exactly **1,000,000,000** non-padding FineWeb-Edu training tokens per run across an **8x AMD Instinct MI300X** cluster, for CauchyLift v0.3 and every frozen baseline (AdamW, Muon) across three confirmatory seeds (`[42, 43, 44]`). Produce complete, resumable, auditable evidence without post-hoc hyperparameter tuning.

## Required work

1. **Preflight Verification:**
   - Verify git commit, protocol hashes (`phase7_125m_protocol.json`), dataset revision, tokenizer, trainable parameter count (125M $\pm 2\%$), environment, RCCL health across all 8 GPUs, free disk space, and VRAM margins.
   - Refuse to execute if protocol drift is detected. Record preflight diagnostics in `artifacts/phase7/preflight.json`.
2. **Distributed Multi-GPU Orchestration:**
   - Launch runs using `torchrun --nproc_per_node=8`.
   - Ensure DDP all-reduce correctly aggregates gradients across ranks before the optimizer step.
   - Ensure CauchyLift v0.3 applies bitwise-identical state-free parameter updates across all 8 ranks without divergence.
3. **Execution of Frozen Suite:**
   - Execute all frozen optimizers (CauchyLift v0.3, AdamW, Muon) across seeds `[42, 43, 44]` in the randomized order registered in Phase 6.
   - Present the exact identical 1B-token stream to every optimizer for a given seed.
   - Do not alter learning rates, warmup, batch size, radius, or stopping rules after inspecting intermediate outcomes.
4. **Telemetry & Monitoring:**
   - Monitor training loss, validation loss on separate FineWeb-Edu validation partition, throughput (tokens/sec), MFU, optimizer-step time, peak memory per rank, loss spikes, gradient-update alignment, and stable rank.
   - Maintain compact structured telemetry (`metrics.jsonl`) and summary logs. Monitoring overhead must remain $< 1\%$ of compute time.
5. **Fault Tolerance & Resumption:**
   - Atomic checkpoints every 250M tokens.
   - In case of a transient GPU or node interruption, resume deterministically from the last valid checkpoint and exact dataset cursor.
   - An OOM must be repaired only by the preregistered equivalent micro-batch / gradient accumulation setting. Never skip a seed.
6. **Completion Verification:**
   - Verify that each successful run consumed exactly 1,000,000,000 non-padding tokens.
   - Verify that validation tokens never contaminated the training stream.
   - Compute checksums over logs, checkpoints, and run metrics.

## Gate

Phase 7 passes only if:

- every frozen optimizer and seed is complete or accounted for under the preregistered failure rule;
- every successful run consumed exactly 1B FineWeb-Edu training tokens on the 125M model across the 8x MI300X cluster;
- paired token streams, validation isolation, and protocol hashes validate with zero discrepancy;
- no unresolved NaN, loss spike, or inter-rank drift remains;
- raw telemetry artifacts are sufficient for the final cross-scale analysis.

Write the standard Phase 7 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase7.json`). Commit and push without force. Do not launch Phase 8 in this session.
