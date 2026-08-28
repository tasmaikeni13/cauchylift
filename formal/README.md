# Lean proof artifact

This directory pins Lean 4.19.0 and mathlib 4.19.0. Run:

```bash
lake update
lake build
```

## Machine-checked scope

- nonnegativity of cotransverse energy from row/column energy bounds;
- the normalized union/complement inequality \(h_{ij}\ge1-a_{ij}\);
- reciprocal and closing algebra behind the \(1/\sqrt3\) angle theorem;
- scale homogeneity and sign alignment of the raw field;
- rank-one cotransverse-energy factorization;
- the 2×2 Cauchy determinant identity and its nondegeneracy;
- scalar bookkeeping for the generic stationarity bound.
- the deterministic sample/noise alignment inequality underlying the conditional stochastic result;
- the exact cubic two-mode gradient-ratio recurrence and its two-step amplification identity.

## Deliberately not claimed as fully formalized

- construction of arbitrary finite matrices and all row/column sums inside Lean;
- the general \(m\times n\) Cauchy determinant product formula;
- the analytic definition of \(L\)-smoothness and the full telescoping theorem;
- measure-theoretic expectations and probability, floating-point behavior, GPU cost, or empirical performance.

The stochastic paper theorem applies expectation to the pointwise lemma under explicit integrability assumptions. Lean checks the deterministic inequality, not the probability-space construction. The mechanism file checks the exact-line orthogonality algebra; the quadratic interpretation remains in the paper.

The boundary is mirrored in `research/proof_audit.md`. A successful `lake build` proves only the statements present in the Lean source; it is not cited as evidence for unformalized prose.
