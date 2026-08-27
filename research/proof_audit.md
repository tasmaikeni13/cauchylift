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
| P8 | Generic exact output rank is \(\min(m,n)\) | Complete via classical Cauchy determinant identity | Only 2×2 nondegeneracy machine checked | Distinct squared factor magnitudes | Algebraic rank, not stable rank |
| P9 | Linear work and zero persistent state | Operation audit | Not a theorem target | Implementation graph | Abstract arithmetic model |

## Assumptions that must remain visible

- The convergence result uses exact gradients. It is not an Adam-style stochastic convergence theorem.
- The step is normalized to a fixed layer radius and uses a prescribed scalar schedule.
- The generic rank theorem assumes distinct squared entries in both rank-one factors and nonzero denominators.
- Algebraic rank says nothing about singular-value mass; the numerical probe is deliberately adverse to overinterpretation.
- GPU friendliness is inferred from the operation graph, not measured.

## Known proof gaps

1. No unconditional stochastic convergence theorem.
2. No useful stable-rank lower bound; sampled evidence argues against expecting one in general.
3. No architecture-aware width/depth scaling theorem.
4. No continuity modulus at the one-sparse projective boundary in finite precision.
5. No rate improvement over first-order lower bounds; the theorem is a safety result, not an acceleration theorem.
