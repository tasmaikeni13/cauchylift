# Phase 6 Prompt — Scaling Pilot, 8x MI300X Orchestration, and Dual Preregistration

Work autonomously in the CauchyLift repository and complete Phase 6. Read `phases/README.md` and require PASS handoffs through Phase 5. This phase establishes multi-GPU distributed orchestration and freezes the final confirmatory protocols for 125M (1B tokens) and 350M (3B tokens) pretraining.

## Objective

Verify distributed scaling across an 8x AMD Instinct MI300X cluster (PyTorch DDP / `torchrun`), validate multi-GPU gradient all-reduction and identical parameter updates across ranks, determine optimal micro-batch and gradient accumulation settings, and commit immutable protocols for:
1. Confirmatory **125M-parameter** decoder-only Transformer trained on **1,000,000,000** FineWeb-Edu tokens per run.
2. Flagship **350M-parameter** decoder-only Transformer trained on **3,000,000,000** FineWeb-Edu tokens per run.

## Required Work

1. **Multi-GPU Orchestration on 8x MI300X:**
   - Integrate PyTorch DDP (`torchrun --nproc_per_node=8`) with ROCm RCCL.
   - Verify that DDP all-reduce aggregates gradients across ranks before the CauchyLift optimizer step.
   - Ensure CauchyLift applies bitwise-identical parameter updates across all 8 ranks with zero drift.
2. **Throughput & MFU Scaling:**
   - Benchmark throughput (tokens/sec) and Model FLOPs Utilization (MFU) scaling across 1, 2, 4, and 8 MI300X GPUs.
   - Optimize sequence length (2048 to 4096), micro-batch size, and gradient accumulation to saturate the 304 Compute Units of each MI300X.
3. **Hyperparameter Grids & Preregistration:**
   - Select and freeze optimal learning rate grids, warmup steps (10%), and cosine decay schedules for CauchyLift, AdamW, and Muon using equal-budget pilot sweeps.
   - Freeze random seeds: `[42, 43, 44]`.
   - Write immutable protocol specifications: `experiments/protocols/phase7_125m_protocol.json` and `experiments/protocols/phase8_350m_protocol.json`.
4. **Resource Verification:**
   - Verify local NVMe storage margin for token caches and atomic checkpoints.
   - Measure time-per-step to confirm 1B-token run duration (~15–25 min on 8x MI300X) and 3B-token run duration (~35–50 min on 8x MI300X).

## Gate

Phase 6 passes only if:
- Multi-GPU DDP scaling on 8x MI300X is verified with zero inter-rank divergence;
- Exact 125M (1B tokens) and 350M (3B tokens) model configs, data splits, and hyperparameters are frozen;
- SHA256 checksums of protocol files are committed;
- Compute and memory resource budgets are measured and confirmed.

Write the standard Phase 6 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase6.json`). Commit without force. Do not start Phase 7 in this session.
