# Phase 4 Prompt — High-Performance Decoder-Only Transformer and Token Data System

Work autonomously in the CauchyLift repository and complete Phase 4. Read `phases/README.md` and require a PASS handoff from Phase 3. This phase constructs the training architecture, data pipeline, and training engine.

## Objective

Build, verify, and smoke-test the complete training stack on AMD Instinct MI300X:
1. Decoder-only Transformer model with FlashAttention (ROCm SDPA), SwiGLU, RMSNorm, RoPE, and tied embeddings.
2. High-throughput memory-mapped binary token data pipeline for FineWeb-Edu.
3. Training engine with BF16 mixed precision, `torch.compile` Inductor integration, gradient accumulation, and telemetry logging.

## Required Work

1. **Transformer Architecture (`cauchylift/models/`):**
   - Configurable decoder-only Transformer:
     - Pre-RMSNorm and SwiGLU activations.
     - Rotary Position Embeddings (RoPE).
     - Tied input and output embedding weights.
     - FlashAttention via ROCm PyTorch `scaled_dot_product_attention`.
   - Dimension specifications: 125M parameter tier ($L=12, H=768, A=12, d_{\text{ffn}}=2048$, $V=50257$, tied params $\approx 123.55\text{M}$) and 350M parameter tier ($L=24, H=1024, A=16, d_{\text{ffn}}=2730$, tied params $\approx 348\text{M}$).
2. **Dataset Pipeline (`cauchylift/data/`):**
   - Pre-tokenized binary token streaming (`np.memmap` / `torch.from_file`) for FineWeb-Edu.
   - Non-overlapping training and validation partitions.
   - Deterministic cursor resumption and index shuffling.
3. **Training & Checkpointing Engine (`cauchylift/train/`):**
   - PyTorch `torch.compile` Inductor support for kernel fusion.
   - Mixed precision training (BF16 autocast).
   - Atomic, crash-resilient checkpointing saving model weights, optimizer state, and step telemetry.
   - JSONL metrics logging tracking step times, optimizer latency, throughput (tokens/sec), gradient norms, and validation perplexity.
4. **Smoke Testing & Overfit Verification:**
   - Verify that the model overfits a synthetic single-batch task to near-zero loss.
   - Verify that FlashAttention and `torch.compile` execute with zero compilation errors on ROCm 10.0 (`gfx942`).

## Gate

Phase 4 passes only if:
- Single-batch overfit achieves $< 0.05$ loss within 100 steps;
- Memory-mapped data loader achieves $> 1,000,000$ tokens/sec streaming bandwidth;
- ROCm FlashAttention executes cleanly with zero NaNs;
- Checkpoint saving and resumption restore bitwise identical loss trajectories.

Write the standard Phase 4 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase4.json`). Commit without force. Do not start Phase 5 in this session.
