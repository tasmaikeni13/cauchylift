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

## Deliberately not claimed as fully formalized

- construction of arbitrary finite matrices and all row/column sums inside Lean;
- the general \(m\times n\) Cauchy determinant product formula;
- the analytic definition of \(L\)-smoothness and the full telescoping theorem;
- floating-point behavior, GPU cost, stochastic convergence, or empirical performance.

The boundary is mirrored in `research/proof_audit.md`. A successful `lake build` proves only the statements present in the Lean source; it is not cited as evidence for unformalized prose.
