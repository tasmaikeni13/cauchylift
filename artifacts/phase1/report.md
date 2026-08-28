# Phase 1 adversarial novelty and mathematical audit

Run ID: `phase1-20260828T102240Z`

Audit date: 2026-08-28 UTC

Input commit: `0038f39`

Scope: theory, novelty, deterministic diagnostics, and Lean; no training loop and no GPU workload

## Decision summary

The cotransverse rational field survives Phase 1 as a scoped-search-distinct theory hypothesis. One real specification defect was found and repaired: the generic rank-lift corollary previously required only pairwise-distinct nonzero squared magnitudes, which still allowed zero factor coordinates. A zero coordinate makes the corresponding diagonal factor singular and can prevent full output rank. The theorem now requires every factor entry to be nonzero, and the counterexample remains checked in the adversarial artifact.

No searched paper, proceeding, preprint, thesis result, searchable patent record, or public optimizer implementation contained the defining normalized field

\[
G_{ij}\big/\left[(S-r_i)+(S-c_j)\right].
\]

This is a finite result for the query ledger through 2026-08-28, not an absolute historical-novelty claim.

## Independent statement audit

| Object | Audit result | Edge conditions and evidence |
|---|---|---|
| Exact energy | Correct | (E_{ij}=(S-r_i)+(S-c_j)\ge0). At a nonzero active entry it is zero exactly when the matrix is one-sparse at that entry. |
| Projective boundary | Correct and total | The regularized ray converges to the signed active cell. `adversarial_audit.json` checks scalars and one-sparse rectangular shapes. Zero maps to zero. |
| Scale and sign | Correct | Squared energies are degree two, the raw field is degree minus one, and normalization gives positive-scale invariance and oddness. |
| Transpose/permutations | Correct | Transpose exchanges row and column complements; permutations only relabel them. One-row/one-column transpose is an explicit regression. |
| Fixed radius and sign | Correct | Normalization assigns Frobenius radius \(\sqrt{\min(m,n)}\); nonzero entries preserve signs. |
| Angle theorem | Correct but conservative | The weighted proof closes at (B\le A+A^2\le3A^2). Equality at (1/\sqrt3) is impossible: away from the boundary every active cell has (h_{ij}<2), hence (A>1/2); the boundary cosine is one. |
| Smooth deterministic convergence | Correct under stated hypotheses | It needs exact gradients, (L\)-smoothness, a lower bound, a constant prescribed step, and either nonzero gradients through the horizon or immediate stationarity. It is a safety rate, not acceleration. |
| Cauchy factorization | Correct | Direct substitution for a rank-one outer product yields the additive Cauchy kernel. |
| Generic algebraic rank lift | Repaired | Every factor entry must be nonzero, squared magnitudes must be pairwise distinct within each factor, and denominators must be nonzero. The old wording fails on (u=(1,0)^\top,v=(1,2)^\top). |
| Complexity/state | Correct in the abstract model | The field uses linear arithmetic, marginal reductions, pointwise operations, scalar normalization, and no persistent optimizer state. No wall-clock statement follows. |

## Adversarial analysis

The new deterministic audit covers zero gradients, scalars, one-sparse and nearly one-sparse matrices, one-row and one-column shapes, repeated marginal energies, approximately 616 orders of input dynamic range, FP16/BF16-representable values, rank-deficient inputs, and the zero-factor rank counterexample. The existing quadratic suite retains axis-aligned and rotated anisotropic cases at Hessian conditions (10^2) and (10^4). The hard rotated and scheduled results remain negative and visible.

Unbiased stochastic gradients are not enough. The retained scalar distribution (g=10) with probability (0.1), (g=-0.5) with probability (0.9) has mean (0.55>0) but expected projective direction (-0.8), giving negative alignment (-0.44). Phase 2 must therefore use explicit noise assumptions rather than importing the deterministic theorem.

## Novelty and composition audit

The refresh searched direct and rearranged formula fragments, row/column complement language, leave-one-out terminology, Cauchy kernels, rational gradient fields, patents, and source-code patterns. It also refreshed the citation neighborhood around historical moments, Adafactor, Shampoo/SOAP, SinkGD, Scion, SWAN, structured Fisher rules, Muon/polar methods, rotations, low-rank methods, symmetry-compatible updates, and 2026 matrix-optimization proceedings.

The closest operational family remains one-pass row/column reduction methods. A generic diagonal representation (P(G)G) is non-discriminative. The tested stronger decompositions did not match: the additive reciprocal is generically not a left diagonal scaling followed by a right diagonal scaling; it is not a row/column balancing fixed point; it is not the LMO of a fixed gradient-independent norm; and it does not compute a polar, whitening, covariance, rotation, or historical-moment module.

## Hard-exclusion audit

The retained version has no momentum, moments, sign branch, clipping, rotation, whitening, polar iteration, low-rank projection, covariance/Fisher approximation, matrix-scaling fixed point, norm-ball LMO, temporal extrapolation, baseline mixture, or safety fallback. Its arithmetic graph remains linear in parameter count with no persistent optimizer state. The core therefore still meets every hard exclusion in `research/research_contract.md`.

## Reproducibility evidence

The following gates passed from a clean checkout plus the Phase 1 changes:

- Python compilation for every analysis module;
- 5,000 randomized algebraic property cases;
- all 16 quadratic-suite trials with negative results retained;
- 200-sample rank/stable-rank probe;
- 200-trial rejected-candidate probe;
- the new adversarial boundary/shape/precision/noise audit;
- the complete declared Lean target with Lean 4.19.0 and pinned mathlib.

Exact commands are in `commands.log`; decisive hashes and environment data are in `manifest.json`.

## Gate result

**PASS** — every active theorem has a correct written proof after the explicit generic-rank hypothesis repair; declared Lean targets build; exact semantics cover zero, scalar, vector, and matrix shapes needed by the current theory; the refreshed, dated collision search found no exact or operationally equivalent primitive within its recorded scope; negative cases remain visible; and the primitive continues to satisfy every hard exclusion. Phase 2 is authorized by this committed handoff.
