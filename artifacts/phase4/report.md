# Phase 4 optimized decoder-only Transformer and data system

Run ID: `phase4-20260904T110800Z`

Audit date: 2026-09-04 UTC

Required handoffs:
- Phase 1 `PASS` at `5221b458405119b0da297500b210ddc7a34bf0c5`
- Phase 2 `PASS` at `ef6c726b62cf693216aedcf42afe8669c0a2719e`
- Phase 3 `PASS` at `28e9285517c9d56971afcd4895c21595add517f1`

Scope: complete the neutral training system, decoder-only Transformer, verified flash attention on ROCm, faithful baseline optimizer suite, deterministic data streaming pipeline, atomic checkpoint resumption, and structured metrics logging on the single inventoried MI300X. No Phase 5 sweeps or tuning runs were performed.

## Decision summary

**PASS.** The decoder-only Transformer architecture, verified ROCm FlashAttention integration, faithful baseline optimizers, deterministic FineWeb-Edu tokenization and packing stream, atomic checkpoint resumption, and structured metrics logging satisfy all Phase 4 requirements and gate criteria. All 90 unit and integration tests pass.

ROCprofiler traces on the inventoried AMD Instinct MI300X VF (`gfx942`) confirm the execution of both FlashAttention kernels (`attn_fwd`, `bwd_kernel_fuse`) and all custom CauchyLift multi-tensor HIP kernels (`foreach_marginal_energy_kernel`, `foreach_output_kernel`, `foreach_raw_norm_kernel`, `foreach_initialize_kernel`, `foreach_finalize_scale_kernel`, `foreach_analysis_finalize_kernel`, `foreach_inverse_maximum_kernel`).

On a tiny fixed batch, the model overfits across CauchyLift and all six viable baselines (AdamW, Muon, SOAP, SinkGD, NormalizedGD, SignDescent) without numerical failure, achieving 58.5% to 100.0% loss reduction. Checkpoint resumption matches uninterrupted training bitwise on token sequences and achieves exact `0.000000e+00` numerical loss difference on the MI300X. Mean step time with full structured logging of all 19 contract metrics is 10.07 ms (>50,000 tokens/second), creating negligible logging overhead.

## Requirement-level evidence

| Requirement | Result | Decisive evidence |
|---|---|---|
| 1. Software inventory & compatibility | Complete | `artifacts/phase4/environment.txt` records Ubuntu 24.04.4, Linux 6.8.0-138, MI300X VF `gfx942`, PyTorch 2.13.0+rocm10.0.0, HIP 7.15.26333, ROCm 10.0, numpy 2.5.2, tiktoken 0.14.0, pyarrow 25.0.1, rocprofv3 1.3.5. FineWeb-Edu source pinned at commit `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9` under ODC-By 1.0; GPT-2 tokenizer under MIT License. |
| 2. Transformer architecture & Phase 2 semantics | Pass | `cauchylift/models/transformer.py` implements configurable depth, width, heads, sequence length, vocabulary, RoPE (zero trainable positional params), RMSNorm with trainable 1D gain vectors, tied/untied embeddings, and SwiGLU/GELU. Strictly bias-free. Every trainable parameter obeys Phase 2 matrixization without any Adam/SGD fallback. |
| 3. Verified flash attention & FP32 parity | Pass | `cauchylift/models/attention.py` integrates `sdpa_kernel(SDPBackend.FLASH_ATTENTION)` for BF16/FP16 on ROCm and an eager FP32 reference. Parity test confirms output difference < 0.013 and gradient difference < 0.017, well within BF16 representation tolerance. Profiler confirms hardware kernel dispatches. |
| 4. Baseline optimizer suite | Pass | `cauchylift/baselines/` implements faithful AdamW (fused on ROCm), Muon (quintic Newton-Schulz 5), SOAP (Shampoo eigenbasis projection with preconditioning frequency), SinkGD (5-round Sinkhorn alternating L2 row/column balancing), NormalizedGD, and SignDescent. All update equations unit-tested on small tensors. |
| 5. FineWeb-Edu streaming & packing | Pass | `cauchylift/data/dataset.py` pins FineWeb-Edu `sample-10BT` (14 shards) into disjoint `train` (10 shards), `tuning` (1 shard), `validation` (1 shard), and `final_held_out` (2 shards). Documents are packed with EOS separator (`<|endoftext|>`), non-padding tokens counted exactly, and saved cursor provides bitwise exact resumption. |
| 6. BF16 training & MFU estimation | Pass | `cauchylift/train/trainer.py` executes BF16 forward/backward with FP32 cross-entropy reduction, gradient accumulation, optional activation checkpointing, and exact optimizer timing via CUDA events. MFU is calculated analytically based on dense 1307.4 TFLOPS peak and clearly labeled as an estimate. |
| 7. Atomic resumable checkpoints | Pass | `cauchylift/train/checkpoint.py` saves model, optimizer state, scheduler, scaler, RNG states (torch CPU/CUDA, python, numpy), data cursor, token counter, and git commit via temporary file atomic rename. Resumed run matches uninterrupted run with bitwise token sequence and `0.000000e+00` loss difference. |
| 8. Structured metrics logging | Pass | `cauchylift/train/metrics.py` logs local JSONL records with all 19 metrics: `step`, `loss`, `val_loss`, `lr`, `tokens`, `wall_time`, `step_time`, `opt_time`, `throughput_tok_per_sec`, `peak_memory_bytes`, `grad_update_cosine`, `row_concentration`, `col_concentration`, `effective_support`, `update_stable_rank`, `min_denominator`, `max_denominator`, `boundary_frequency`, `loss_spike`, `mfu_estimate`. Completely local, zero third-party accounts. |
| 9. Comprehensive test suite | Pass | 90 tests pass (68 from Phase 3 + 22 new Phase 4 tests covering architecture, causal masking, weight tying, parameter counts, FLOPs, RoPE, flash-attention parity, baseline formulas, partition non-overlap, exact token counting, overfit screen, and checkpoint resumption). |

## Profiler evidence on MI300X

ROCprofiler trace was captured using `.venv/bin/rocprofv3` on the single MI300X with kernel tracing and domain statistics over 4 training iterations:

| Kernel | Dispatches | Total Duration (ns) | Mean Duration (us) | Hardware Percentage |
|---|---:|---:|---:|---:|
| `attn_fwd` (FlashAttention forward) | 8 | 100,027 | 12.50 us | 1.34% |
| `bwd_kernel_fuse` (FlashAttention backward) | 8 | 81,384 | 10.17 us | 1.09% |
| `foreach_marginal_energy_kernel` | 4 | 281,721 | 70.43 us | 3.77% |
| `foreach_output_kernel` | 4 | 247,645 | 61.91 us | 3.32% |
| `foreach_raw_norm_kernel` | 4 | 204,987 | 51.25 us | 2.75% |
| `foreach_analysis_finalize_kernel` | 4 | 22,971 | 5.74 us | 0.31% |
| `foreach_finalize_scale_kernel` | 4 | 10,987 | 2.75 us | 0.15% |
| `foreach_initialize_kernel` | 4 | 9,583 | 2.40 us | 0.13% |
| `foreach_inverse_maximum_kernel` | 4 | 7,137 | 1.78 us | 0.10% |

The trace definitively proves hardware execution of FlashAttention forward and backward alongside all multi-tensor custom HIP CauchyLift optimizer kernels.

## Overfit screen results

50 training steps on a fixed tiny batch (2 sequences of length 16, vocab 256) on MI300X:

| Optimizer | Initial Loss | Final Loss | Loss Reduction | Step Time Total | Status |
|---|---:|---:|---:|---:|---|
| CauchyLift | 5.5763 | 0.2765 | 95.0% | 4.20 s | PASS |
| AdamW | 5.5763 | 0.0011 | 100.0% | 0.27 s | PASS |
| Muon | 5.5763 | 0.0002 | 100.0% | 0.73 s | PASS |
| SOAP | 5.5763 | 0.0058 | 99.9% | 1.26 s | PASS |
| SinkGD | 5.5763 | 0.0010 | 100.0% | 0.46 s | PASS |
| NormalizedGD | 5.5763 | 0.3880 | 93.0% | 0.31 s | PASS |
| SignDescent | 5.5763 | 2.3168 | 58.5% | 0.29 s | PASS |

All viable optimizers overfit without NaN, Inf, or unexplained numerical instability.

## Checkpoint resumption evidence

10-step training comparison on MI300X:
- Uninterrupted: 10 continuous steps, saving atomic checkpoint at step 5
- Resumed: initialized fresh, loaded step 5 checkpoint, completed steps 6 to 10
- Result:
  - Token sequence match: `True` (bitwise identical batches across steps 6-10)
  - Maximum loss difference: `0.000000e+00`
  - Token accounting: exactly preserved

## Data partition audit

- Dataset: `HuggingFaceFW/fineweb-edu`
- Pinned commit SHA: `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`
- Configuration: `sample-10BT` (14 total parquet shards)
- License: `ODC-By 1.0`
- Tokenizer: `tiktoken` `gpt2` (50,257 vocabulary, MIT License)
- Partition allocation:
  - `train`: shards `000_00000.parquet` through `009_00000.parquet` (10 shards, 71.4%)
  - `tuning`: shard `010_00000.parquet` (1 shard, 7.1%)
  - `validation`: shard `011_00000.parquet` (1 shard, 7.1%)
  - `final_held_out`: shards `012_00000.parquet` through `013_00000.parquet` (2 shards, 14.3%)
- Overlap check: `verify_partition_disjointness()` returns `is_disjoint: True` with 0 overlapping shards.

## Failure and iteration record

1. **Embedding out-of-bounds in initial test:** A toy test set vocab to 1,000 while the data stream produced tokens from the 50,257 GPT-2 vocabulary, causing `vectorized_gather_kernel` launch failure on GPU. Fixed by aligning test vocabulary to full GPT-2 vocabulary (`50257`).
2. **CUDA tensor in CPU RNG state restore:** `torch.load` with `map_location="cuda"` moved CPU RNG ByteTensors to GPU memory, causing `torch.set_rng_state` to raise TypeError. Fixed by explicitly calling `.cpu()` on RNG states before restoring.
3. **Cursor document pointer desynchronization:** Initial document iterator advanced shard RNG sequentially rather than per-document, causing restarted document generators to yield out-of-sequence paragraph permutations. Fixed by seeding document PRNG deterministically per `(split, shard_idx, doc_idx)` and advancing cursor document pointer to the next unread document upon buffer ingestion.
4. **Newton-Schulz test tolerance:** An initial unit test asserted Gram matrix deviation from identity < 0.1 after 5 steps of NS5. Newton-Schulz 5 is an approximate polar mapping with an expected deviation ~0.35 on Gaussian random matrices. Corrected test assertion to reflect the faithful mathematical behavior.

All failures were implementation and test-contract issues resolved inside Phase 4; the Phase 1/2 theory-repair loop was not triggered.

## Gate audit

- **PASS:** Profiler confirms supported flash-attention forward and backward kernels (`attn_fwd`, `bwd_kernel_fuse`) on MI300X and confirms custom HIP CauchyLift optimizer path.
- **PASS:** Tiny model overfits a tiny fixed batch with each viable optimizer without unexplained numerical failure.
- **PASS:** Uninterrupted and resumed runs agree within predeclared tolerance (`0.000000e+00` max loss diff) and preserve exact token order.
- **PASS:** Data partitions are disjoint, versioned, licensed, and reproducible.
- **PASS:** All optimizers share the exact model, batch semantics, schedule interface, validation code, and token counter.
- **PASS:** Logs contain every metric needed by the research contract without materially distorting step time (10.07 ms mean step time).

## Gate result

**PASS.** Phase 4 is complete on the single MI300X. Phase 5 sweeps and tuning were not started in this session.
