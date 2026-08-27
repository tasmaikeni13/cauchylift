# Phase 4 prompt — optimized decoder-only Transformer and data system

Work autonomously in the CauchyLift repository and complete Phase 4. Read phases/README.md and require PASS handoffs through Phase 3. Build a neutral training system that can compare optimizers without changing the model, data, or token accounting.

## Objective

Create reproducible decoder-only Transformer training scripts for ROCm and the MI300X, with verified flash attention, the native HIP optimizer path, efficient BF16 training, deterministic data order, resumable checkpoints, and enough instrumentation for fair optimizer science. Run smoke tests only; do not begin the large comparison.

## Required work

1. Re-inventory software compatibility and consult current primary documentation for PyTorch ROCm, scaled-dot-product or flash attention, HIP extensions, and the selected FineWeb-Edu source. Pin working versions and record the dataset revision, license, configuration, and tokenizer license.
2. Implement a compact GPT-style decoder-only model with configurable depth, width, heads, sequence length, vocabulary, RoPE or another frozen positional choice, normalization, tied or untied embeddings, and activation. Keep the architecture identical for all optimizers. Ensure its trainable shapes obey Phase 2 semantics without an Adam/SGD fallback.
3. Use an attention implementation that is actually flash or memory efficient on this ROCm stack. Verify backend selection through profiler traces and compare outputs and gradients to a small eager FP32 reference. Do not infer flash execution from an API name. Fall back only as an explicitly failed engineering case, not as a PASS.
4. Integrate the Phase 3 PyTorch and HIP CauchyLift paths behind one exact configuration. Integrate faithful AdamW, Muon, SOAP, SinkGD, normalized-gradient, and sign-descent baselines from primary definitions or maintained source. Unit-test update equations on small tensors. Do not normalize away meaningful baseline behavior.
5. Build a deterministic streaming and tokenization pipeline for FineWeb-Edu. Pin the source revision; record document IDs or shard hashes; create disjoint train, tuning, validation, and final-held-out partitions; define BOS/EOS and document packing; count non-padding training tokens exactly; and make the stream restartable from a saved cursor. Never commit the dataset cache.
6. Implement BF16 training with FP32-sensitive reductions, gradient accumulation, optional activation checkpointing, compile settings only when verified, and single-GPU execution. Record model FLOPs utilization estimates carefully and label estimates as estimates.
7. Make checkpoints atomic and resumable. Save model, optimizer state when present, scheduler, scaler if any, RNG states, data cursor, token count, run configuration, source commit, and environment fingerprint. A resumed run must match an uninterrupted deterministic micro-run within a declared tolerance.
8. Log local structured records for loss, validation loss, learning rate, tokens, wall time, optimizer-only time, throughput, peak memory, gradient/update cosine, row and column concentration, effective support, update stable rank, denominator extremes, boundary frequency, and loss spikes. Do not require a third-party account.
9. Add tests for causal masking, weight tying, parameter counts, tokenizer packing, data partition non-overlap, exact token budgets, baseline equations, checkpoint resume, flash-attention parity, and short overfit behavior. Run tiny CPU tests and short MI300X smoke tests.

## Gate

Phase 4 passes only if:

- the profiler confirms a supported flash-attention or equivalent memory-efficient kernel on MI300X and confirms the custom HIP optimizer path;
- a tiny model overfits a tiny fixed batch with each viable optimizer without unexplained numerical failure;
- uninterrupted and resumed runs agree within the predeclared tolerance and preserve exact token order;
- data partitions are disjoint, versioned, licensed, and reproducible;
- all optimizers share the exact model, batch semantics, schedule interface, validation code, and token counter;
- logs contain every metric needed by the research contract without materially distorting step time.

If a failure exposes faulty optimizer mathematics, enter the shared theory-repair loop and invalidate Phases 3 and 4 as needed. Write the standard Phase 4 artifacts. Commit and push validated source and small artifacts without force. Do not start tuning or large training in this session.
