# Phase 9 Prompt — Cross-Scale Statistical Analysis, Reproducibility Audit, and Publishable Paper

Work autonomously in the CauchyLift repository and complete Phase 9. Read `phases/README.md` and require a PASS handoff from Phase 8. Analyze the empirical evidence across all phases and finalize a publication-grade manuscript.

## Objective

Synthesize all mathematical, formal, systems, and empirical evidence across the entire project into a submission-ready research paper covering:
1. First-principles derivation of the Additive Fiber RMS Cauchy Operator ($D_{ij} = \text{RMS}(M_{i,:}) + \text{RMS}(M_{:,j})$) and longest-fiber radius scaling ($\rho = \sqrt{\max(m, n)}$).
2. Theoretical guarantees: degree-0 scale invariance, strict coordinate bounds, and positive descent alignment.
3. Systems architecture: native multi-tensor HIP kernels on AMD Instinct MI300X with single-state memory overhead (50% less memory than AdamW) and sub-millisecond execution ($>10\times$ faster than Muon).
4. Multi-scale pretraining results: small/medium screen (Phase 5), 125M on 1B tokens (Phase 7), and 350M on 3B tokens of FineWeb-Edu (Phase 8).
5. Empirical scaling laws, tokens-to-target comparisons, perplexity tables, memory savings, and reproducibility kits.

## Required Work

1. **Telemetry & Artifact Validation:**
   - Validate run indices, SHA256 hashes, token counts, data partitions, seeds, and configs across all phases (Phase 1 through Phase 8).
   - Recompute decisive summary statistics from raw structured logs using standalone scripts.
2. **Statistical & Scaling Law Analysis:**
   - Perform paired statistical tests for tokens-to-target, final validation loss, perplexity, wall-clock throughput, and peak memory between CauchyLift, AdamW, and Muon.
   - Fit cross-scale power-law scaling exponents $L(N, C)$ connecting small/medium models, 125M, and 350M.
   - Characterize the exact memory-reduction frontier (1 state tensor vs AdamW's 2 state tensors).
3. **Primary Literature & Related-Work Audit:**
   - Refresh the literature audit through the analysis date using primary sources and mathematical comparisons (against AdamW, Adafactor, Shampoo, SOAP, Muon).
   - Document the uniqueness of instantaneous spatial fiber RMS normalization over momentum velocity.
4. **Manuscript Authoring (`paper/paper.md`):**
   - Refine the complete, canonical research manuscript:
     - Abstract, Introduction, Mathematical Foundation, Systems Architecture, Experimental Protocols, Results, Discussion, and Conclusion.
     - Ensure clear separation of proved theorems, machine-checked lemmas, and measured empirical evidence.
5. **Typesetting & Reproducibility:**
   - Generate publication-quality vector plots (paired loss curves, scaling surfaces, memory scaling, and kernel microbenchmarks).
   - Update `README.md`, `REPRODUCIBILITY.md`, `CITATION.cff`, and the evidence ledger.
   - Provide clean, deterministic one-line commands for verification, unit testing, and result replication.

## Gate

Phase 9 passes only if:
- All tables, figures, and scaling curves are reproducible from tracked scripts and checksummed raw logs;
- Manuscript agrees with preregistered protocols and reports all findings honestly;
- Formal Lean 4 proofs, ROCm kernel code, and empirical results are unified;
- The paper is submission-ready for top-tier machine learning venues.

Write the final Phase 9 artifacts (`report.md`, `manifest.json`, `paper/cauchylift_paper.pdf`). Commit without force. Tag the final release upon completion.
