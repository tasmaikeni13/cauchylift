# Proof audit

## Statement inventory

| ID | Statement | Paper proof | Lean status | Dependencies | Scope |
|---|---|---|---|---|---|
| P1 | Cotransverse energy is nonnegative | Complete | Machine checked | Finite sums of squares | All real matrices |
| P2 | Normalized cell field obeys \(h_{ij}\ge1-a_{ij}\) and \(h_{ij}\le2\) | Complete | Machine checked in finite-table abstraction | Row/column union identity | Nonzero matrices |
| P3 | Scale invariance, oddness, transpose and row/column permutation equivariance | Complete | Core scalar identities checked; permutation proof on paper | Homogeneity | Nonzero matrices, projective limit at boundary |
| P4 | Strict descent alignment | Complete | Machine checked for positive denominators; boundary handled separately | Sign preservation | Exact gradient |
| P5 | Cosine is at least \(1/\sqrt3\) | Complete | Finite-weight inequality machine checked | P2 and elementary sum inequalities | Exact gradient |
| P6 | Smooth deterministic stationarity bound \(\min_t\|\nabla f(W_t)\|_F\le\sqrt{6L\Delta/T}\) | Complete | Descent-summation algebra machine checked | L-smoothness, lower bound, chosen step | Full deterministic gradients |
| P7 | Rank-one input induces a Cauchy-kernel factorization | Complete | 2×2 case machine checked | Nonzero factors | Away from one-sparse boundary |
| P8 | Generic exact output rank is \(\min(m,n)\) | Complete via classical Cauchy determinant identity; hypothesis repaired in Phase 1 | Only 2×2 nondegeneracy machine checked | Every factor entry nonzero; squared magnitudes pairwise distinct within each factor | Algebraic rank, not stable rank |
| P9 | Linear work and zero persistent state | Operation audit | Not a theorem target | Implementation graph | Abstract arithmetic model |

## Assumptions that must remain visible

- The convergence result uses exact gradients. It is not an Adam-style stochastic convergence theorem.
- The step is normalized to a fixed layer radius and uses a prescribed scalar schedule.
- The generic rank theorem assumes every entry of both rank-one factors is nonzero, squared entries are pairwise distinct within each factor, and all selected denominators are nonzero. The all-entries-nonzero hypothesis is necessary: \(u=(1,0)\), \(v=(1,2)\) is a retained counterexample to the earlier wording.
- Algebraic rank says nothing about singular-value mass; the numerical probe is deliberately adverse to overinterpretation.
- GPU friendliness is inferred from the operation graph, not measured.

## Known proof gaps

1. No unconditional stochastic convergence theorem.
2. No useful stable-rank lower bound; sampled evidence argues against expecting one in general.
3. No architecture-aware width/depth scaling theorem.
4. No continuity modulus at the one-sparse projective boundary in finite precision.
5. No rate improvement over first-order lower bounds; the theorem is a safety result, not an acceleration theorem.

## Phase 1 repair record

The 2026-08-28 adversarial audit found one false hypothesis boundary, not a false conclusion under generic conditions. The previous phrase “the nonzero squared entries are pairwise distinct” allowed zero coordinates and therefore did not justify preservation of rank by the diagonal factors. The paper now requires every factor entry to be nonzero. `analysis/run_adversarial_audit.py` records the \(2\times2\) zero-row counterexample as a regression. No active theorem was weakened silently.
