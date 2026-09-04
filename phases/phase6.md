# Phase 6 prompt — scaling pilot, 8x MI300X orchestration, and dual preregistration

Work autonomously in the CauchyLift repository and complete Phase 6. Read phases/README.md and require PASS handoffs through Phase 5. This phase uses bounded multi-GPU MI300X time to establish scaling laws and freeze the final confirmatory protocols for both 125M (1B tokens) and 350M (3B tokens) before final held-out runs.

## Objective

Test whether the Phase 5 CauchyLift v0.3 advantage transfers across model scaling, verify 8x MI300X multi-GPU distributed orchestration (PyTorch DDP / `torchrun`), select hyperparameters with equal budgets, estimate compute and storage from measurements, and produce immutable protocols for training:
1. An approximately **125M-parameter** decoder-only Transformer on exactly **1,000,000,000** FineWeb-Edu tokens per confirmatory run.
2. An approximately **350M-parameter** decoder-only Transformer on exactly **3,000,000,000** FineWeb-Edu tokens per flagship run.

## Required work

1. **Commit a pilot protocol before execution:** Define model scales (e.g. 35M and 70M intermediate pilots), token budgets, seeds, tuning partitions, optimizer grids, cosine schedules, global batch size, and stop rules. Do not use the final held-out test partition for tuning.
2. **Multi-GPU Orchestration on 8x MI300X:**
   - Integrate PyTorch Distributed Data Parallel (`torchrun --nproc_per_node=8`) with ROCm RCCL.
   - Verify that CauchyLift v0.3 executes cleanly in DDP (gradients all-reduced across ranks in FP32/BF16, identical state-free update applied across ranks with 0 drift).
   - Benchmark throughput and MFU scaling across 1, 2, 4, and 8 MI300X GPUs with FlashAttention.
3. **Equal-Budget Pilot Runs:**
   - Run equal-budget scaling sweeps for CauchyLift v0.3, AdamW, and Muon. Include SOAP and NormalizedGD controls where informative.
   - Measure validation loss, tokens-to-target, step times, MFU, throughput (tokens/sec), peak memory, gradient-update alignment, and loss spike frequency.
4. **Freeze 125M / 1B Architecture & Data Protocol:**
   - Trainable parameter count: 125M within $\pm 2\%$.
   - Architecture: layers, hidden dim, heads, seq len 2048, SwiGLU / GeLU, tied embeddings, RoPE, RMSNorm.
   - Exact FineWeb-Edu revision and deterministic 1B-token stream.
   - Freeze learning rates, warmup (10%), global batch size (e.g. 512K tokens), checkpoint cadence, and stopping rules.
5. **Freeze 350M / 3B Architecture & Data Protocol:**
   - Trainable parameter count: 350M within $\pm 2\%$ (e.g. 24 layers, 1024 hidden dim, 16 heads, seq len 2048).
   - Exact FineWeb-Edu revision and deterministic 3B-token stream.
   - Freeze learning rates, warmup, global batch size, and evaluation cadence.
6. **Final Optimizer Set & Decision Rules:**
   - Freeze optimizers: CauchyLift v0.3, AdamW, Muon.
   - Use at least three confirmatory seeds per optimizer (`[42, 43, 44]`).
   - Define exact primary decision rules (tokens-to-target, final validation loss, perplexity on held-out FineWeb-Edu, and memory savings).
7. **Resource & Checkpoint Plan:**
   - Verify local disk margin for FineWeb-Edu shards and atomic checkpoints.
   - Verify that each 125M/1B run consumes ~15–25 minutes on 8x MI300X, and each 350M/3B run consumes ~30–45 minutes on 8x MI300X.
8. **Preregistration Artifacts:**
   - Write immutable protocol files under `experiments/protocols/phase7_125m_protocol.json` and `experiments/protocols/phase8_350m_protocol.json`.
   - Record protocol hashes and commit. Add automated validators ensuring zero protocol drift before launching Phases 7 and 8.

## Gate

Phase 6 passes only if:

- multi-GPU scaling on 8x MI300X is verified with zero numerical drift across ranks;
- pilot sweeps preserve the CauchyLift v0.3 advantage and show no new theory, rank, or boundary failure;
- all compared optimizers received identical tuning budgets across pilot scales;
- exact 125M (1B tokens) and 350M (3B tokens) model architectures, FineWeb-Edu data streams, seeds, and hyperparameters are frozen before execution;
- a measured resource plan confirms the complete suite can run safely within available disk and compute budgets;
- preregistration commit and SHA256 hashes are recorded in `artifacts/phase6/report.md`.

Write the standard Phase 6 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase6.json`). Commit and push without force. Do not launch the full 1B or 3B token runs in this session.
