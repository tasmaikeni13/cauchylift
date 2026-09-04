# Phase 9 prompt — cross-scale statistical analysis, reproducibility audit, and publishable paper

Work autonomously in the CauchyLift repository and complete Phase 9. Read phases/README.md and require a PASS handoff from Phase 8. Analyze the frozen evidence from both the 125M / 1B-token and 350M / 3B-token experiments, and turn the full research project into a submission-ready, publication-grade manuscript. Do not submit it to a venue or claim external peer review.

## Objective

Synthesize all mathematical, formal, systems, and empirical evidence across the entire project. Produce a rigorous, publication-ready research paper covering:
1. First-principles derivation of the Additive Fiber RMS Cauchy Kernel ($D_{ij} = \text{RMS}(G_{i,:}) + \text{RMS}(G_{:,j})$) and longest-fiber radius scaling ($\rho = \sqrt{\max(m, n)}$).
2. Machine-checked formal proofs in Lean 4 (degree-0 scale invariance, strict positivity, strict descent alignment, and coordinate magnitude bounds).
3. Native ROCm/HIP multi-tensor kernel implementation on AMD Instinct MI300X with zero persistent optimizer state.
4. Complete small-scale screen results (Phase 5) across 4 workloads and 7 optimizers.
5. Dual-scale pretraining results: 125M on 1B tokens (Phase 7) and 350M on 3B tokens of FineWeb-Edu (Phase 8) on the 8x MI300X cluster.
6. Empirical scaling laws, tokens-to-target comparisons, perplexity tables, memory savings, and failure-proof reproducibility audits.

## Required work

1. **Telemetry & Artifact Validation:**
   - Validate run indices, SHA256 hashes, token counts, data partitions, seeds, and configs across all phases (Phase 1 through Phase 8).
   - Recompute decisive summary statistics from raw structured logs (`runs/`) using a standalone tested script.
2. **Paired Statistical Analysis:**
   - Perform paired statistical tests for tokens-to-target, final validation loss, perplexity, wall-clock throughput, and peak memory between CauchyLift v0.3, AdamW, and Muon.
   - Fit cross-scale power-law scaling exponents $L(N, C)$ connecting Phase 5 (small/medium LM), Phase 7 (125M), and Phase 8 (350M).
   - Characterize the exact memory-reduction frontier (0 bytes persistent state vs AdamW's $2\times$ model parameter memory).
3. **Primary Novelty & Related-Work Audit:**
   - Refresh the scoped literature audit through the analysis date using primary sources and mathematical comparisons (against Adam, RMSprop, Adafactor, Shampoo, SOAP, Muon, NormalizedGD).
   - Formally document the uniqueness of instantaneous spatial fiber RMS normalization versus temporal moment accumulation.
4. **Manuscript Authoring (`paper/paper.md`):**
   - Write the complete, canonical research manuscript:
     - **Abstract:** Motivation, mathematical formulation, formal guarantees, and empirical results on FineWeb-Edu up to 350M / 3B tokens.
     - **Introduction & Vision:** The quest for 2nd-order-like curvature adaptation at 1st-order hardware speed with 0 persistent state.
     - **Mathematical Foundation:** Fiber bundle geometry, additive Cauchy kernel structure, coordinate bounds, and Lean 4 formalization.
     - **Systems Architecture:** Native multi-tensor HIP kernel design, DDP multi-GPU scaling on 8x MI300X, and zero-memory allocation.
     - **Experimental Protocol:** Equal-budget hyperparameter screen, held-out ConvSSM transfer, and FineWeb-Edu pretraining.
     - **Results:** Phase 5 screen (170 runs), Phase 7 (125M / 1B), and Phase 8 (350M / 3B flagship).
     - **Discussion & Limitations:** Exact scope of verified claims, negative results honestly retained, and future research directions.
5. **Typesetting & Figures:**
   - Generate publication-quality vector figures (paired loss curves, scaling surfaces, memory scaling plots, and kernel microbenchmarks).
   - Compile a submission-ready PDF paper and complete BibTeX bibliography. Inspect for broken references, formatting defects, and claim-evidence consistency.
6. **Reproducibility Kit:**
   - Update `README.md`, `REPRODUCIBILITY.md`, `CITATION.cff`, and the evidence ledger.
   - Provide clean, deterministic one-line commands for Lean 4 verification, test suite execution, and result replication.
7. **Claim-to-Evidence Audit:**
   - Construct an exhaustive claim-to-evidence table labeling every theorem as Lean-proved, every benchmark as measured on MI300X, and eliminating marketing exaggeration.

## Gate

Phase 9 passes only if:

- all tables, figures, and scaling curves are reproducible from tracked scripts and checksummed raw logs;
- the manuscript agrees with preregistered protocols and reports all negative findings honestly;
- formal Lean 4 boundaries, ROCm kernel code, and empirical results are fully unified;
- the paper is submission-ready for top-tier machine learning venues;
- repository audit from a clean checkout passes with zero missing dependencies or broken links.

Write the final Phase 9 artifacts (`report.md`, `manifest.json`, `paper/cauchylift_paper.pdf`). Commit and push without force. Tag the final release upon completion.
