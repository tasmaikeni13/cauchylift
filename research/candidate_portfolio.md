# Candidate portfolio and rejection ledger

The final proposal was selected by elimination, not by naming the first unusual formula.

## C1 — Exterior/cofactor area dual

**Idea.** Use 2×2 determinants, cofactors, and an exterior-area potential to condition matrix gradients with local arithmetic.

**Unique prediction.** The singular-value condition map would differ from polar iterations while retaining their rapid conditioning.

**Attack.** Symbolically reduce the induced condition-number map.

**Result.** For \(\kappa>1\), the map becomes

\[
\kappa' = \frac{\kappa(\kappa^2+3)}{3\kappa^2+1}
          = \coth(3\,\operatorname{arccoth}\kappa),
\]

the Halley/polar triple-angle map. The numerical identity is checked in `run_rejection_checks.py`.

**Decision.** Rejected as a disguised polar method.

## C2 — Plaquette cross-ratio inversion

**Idea.** On a 2×2 tile, invert the multiplicative cross-ratio and choose the unique fixed-RMS, sign-preserving representative.

**Unique prediction.** Remove nonseparable row-column anisotropy without Sinkhorn balancing.

**Attack.** Exact-line-search diagonal quadratics across four decades of curvature.

**Result.** 198 of 200 seeded trials failed to reach the target within 1,000 iterations, versus 46 failures for sign descent. The map can ignore a large informative gradient when its opposite product is small.

**Decision.** Rejected. A safety mixture would turn it into a conventional hybrid.

## C3 — Spherical jet continuation

**Idea.** Continue two normalized gradient directions along their great circle using \(d_t=2\langle u_{t-1},u_t\rangle u_t-u_{t-1}\).

**Unique prediction.** Exact one-step prediction when gradient directions move at constant angular velocity.

**Attack.** Compare with optimistic/extrapolated-gradient literature and small curved/ill-conditioned objectives.

**Result.** The operation is close to normalized optimistic extrapolation, and its improvement was inconsistent. Trust gates make it a composite method.

**Decision.** Rejected for proximity and weak mechanism.

## C4 — Outer-product cycle-syndrome repair

**Idea.** Treat rank-one per-example gradient signs as a bipartite cycle code and decode violated four-cycle parities.

**Unique prediction.** Correct independent sign corruption using algebraic redundancy.

**Attack.** Check the structural premise after minibatch aggregation.

**Result.** A minibatch gradient is a sum of outer products and need not obey rank-one cycle parity. Accessing per-example gradients changes the cost model. The proposal also reads as coding theory appended to sign descent.

**Decision.** Rejected.

## C5 — Temporal projective holonomy

**Idea.** Use cross-ratios of two coordinates across two time steps to measure relative gradient growth, then apply the dual holonomy.

**Unique prediction.** Invariance to coordinate rescaling and per-step loss scaling with better damping of stiff coordinates.

**Attack.** Exact-line-search diagonal quadratics and comparison to gradient extrapolation.

**Result.** No consistent advantage; the dual rule often lagged sign descent. It is also adjacent to multiplicative gradient prediction.

**Decision.** Rejected.

## C6 — CauchyLift cotransverse rational field

**Idea.** Couple entry \((i,j)\) only to energy excluded by its row and column:

\[
Z_{ij}=\frac{G_{ij}}{(S-r_i)+(S-c_j)}.
\]

**Unique predictions.** Dimension-independent descent angle; generic Cauchy factorization on rank-one inputs; dense, smooth concentration on energetic row-column intersections; linear reduction-only implementation.

**Attacks completed.** Algebraic proof, 5,000 randomized property checks, exact and scheduled quadratic probes at condition 100 and 10,000, exact-rank and stable-rank probes, and a closest-family literature audit.

**Result.** Survives as a theory hypothesis. It performs strongly on several condition-100 and axis-aligned exact-line probes, but degrades under the scheduled condition-10,000 probe and gains essentially no stable rank in the sampled rank-one cases.

**Decision.** Retained for formalization, not promoted as empirically validated.
