# Phase 1 Prompt — Mathematical Foundation and Scoped Novelty Audit

Work autonomously in the CauchyLift repository and complete Phase 1. Read `phases/README.md` first and obey its state machine. This is a theory and novelty phase: do not create a heavy neural-network training loop and do not spend meaningful GPU time.

## Objective

Audit the mathematical foundation of CauchyLift. Formally verify the Additive Fiber RMS Cauchy operator, the directional momentum-filtered manifold, degree-0 scale invariance, coordinate-wise magnitude bounds, and strict positive descent alignment. Conduct a scoped, dated literature audit confirming novelty relative to prior matrix, coordinate-wise, and polar optimizers.

## Required Work

1. **Mathematical Audit:**
   - Verify the defining equations:
     $$M_t = \beta M_{t-1} + (1 - \beta) G_t$$
     $$D_{ij} = \text{RMS}(M_{i,:}) + \text{RMS}(M_{:,j}) = \sqrt{\frac{1}{n}\sum_{k=1}^n M_{ik}^2} + \sqrt{\frac{1}{m}\sum_{l=1}^m M_{lj}^2}$$
     $$Z_{ij} = \frac{M_{ij}}{D_{ij}}, \qquad U = \sqrt{\max(m, n)} \frac{Z}{\|Z\|_F}$$
     $$W_{t+1} = W_t (1 - \eta \lambda) - \eta U$$
   - Audit proofs of:
     - Degree-0 scale invariance: $U(\alpha M) = U(M)$ for all $\alpha > 0$.
     - Coordinate magnitude bounds: $|Z_{ij}| \le \min(\sqrt{n}, \sqrt{m})$.
     - Strict descent alignment: $\langle M, U(M) \rangle_F > 0$ for all non-zero $M$.
     - Zero and 1-sparse boundary behavior: well-defined continuous projective limit.
2. **Adversarial & Numerical Probes:**
   - Construct symbolic and numerical tests covering: 1-sparse matrices, one-row/one-column vectors, extreme dynamic ranges ($10^{-30}$ to $10^{30}$), rank-deficient matrices, anisotropic quadratics, and varying momentum factors.
3. **Scoped Novelty Audit:**
   - Search literature through the current date across primary papers, preprints, and open-source optimizer implementations.
   - Compare CauchyLift against AdamW, Adafactor, Shampoo, SOAP, Muon, NormalizedGD, and SignSGD.
   - Record exact defining equations and distinction ledgers in `research/closest_work_matrix.md` and `research/search_log.md`.
4. **Formalization (Lean 4):**
   - Verify or extend formal machine-checked proofs in `formal/CauchyLift/` covering scale invariance, denominator strict positivity, and coordinate bounds.

## Gate

Phase 1 passes only if:
- All active theorems have written proofs and matching formal Lean 4 checks;
- Edge-case semantics (0-gradient, 1-sparse, vector/scalar shapes) are rigorously defined;
- No searched source contains the identical defining primitive;
- Novelty audit is scoped, dated, and query-backed in `research/search_log.md`.

Write the standard Phase 1 artifacts (`artifacts/phase1/report.md`, `manifest.json`, `commands.log`, `phases/status/phase1.json`). Commit without force. Do not start Phase 2 in this session.
