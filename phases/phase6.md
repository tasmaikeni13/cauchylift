# Phase 6 Prompt — Scaling Pilot, 16x TPU v4-32 Orchestration, and Dual Preregistration

Work autonomously in the CauchyLift repository and complete Phase 6. Read `phases/README.md` and require PASS handoffs through Phase 5. This phase establishes multi-TPU distributed orchestration and freezes the final confirmatory protocols for 125M (3B tokens) and 350M (3B tokens) pretraining.

## Objective

Verify distributed scaling across a 16x Google Cloud TPU v4 pod slice (Torch-XLA / PJRT), validate multi-core gradient all-reduction and identical parameter updates across ranks, determine optimal micro-batch and gradient accumulation settings, and commit immutable protocols for:
1. Confirmatory **125M-parameter** decoder-only Transformer trained on **3,000,000,000** FineWeb-Edu tokens per run.
2. Flagship **350M-parameter** decoder-only Transformer trained on **3,000,000,000** FineWeb-Edu tokens per run.

## Required Work

1. **Multi-Core Orchestration on 16x TPU v4-32:**
   - Integrate Torch-XLA PJRT distributed execution across all 16 TPU chips.
   - Verify that all-reduce aggregates gradients across cores before the CauchyLift optimizer step.
   - Ensure CauchyLift applies bitwise-identical parameter updates across all ranks with zero drift.
2. **Throughput & MFU Scaling:**
   - Benchmark throughput (tokens/sec) and Model FLOPs Utilization (MFU) scaling across 1, 4, 8, and 16 TPU chips.
   - Optimize sequence length (2048 to 4096), micro-batch size, and gradient accumulation to saturate the VPU/MXU units of each TPU v4 chip.
3. **Hyperparameter Grids & Preregistration:**
   - Select and freeze optimal learning rate grids, warmup steps (10%), and cosine decay schedules for CauchyLift, AdamW, and Muon using equal-budget pilot sweeps.
   - Freeze random seeds: `[42, 43, 44]`.
   - Write immutable protocol specifications: `experiments/protocols/protocol_125m_fineweb.json` and `experiments/protocols/protocol_350m_fineweb.json`.
4. **Resource Verification:**
   - Verify storage margin for token caches and atomic checkpoints.
   - Measure time-per-step to confirm 3B-token run duration on 16x TPU v4-32.

## Gate

Phase 6 passes only if:
- Multi-device scaling on 16x TPU v4-32 is verified with zero inter-rank divergence;
- Exact 125M (3B tokens) and 350M (3B tokens) model configs, data splits, and hyperparameters are frozen;
- SHA256 checksums of protocol files are committed;
- Compute and memory resource budgets are measured and confirmed.

Write the standard Phase 6 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase6.json`). Commit without force. Do not start Phase 7 in this session.
