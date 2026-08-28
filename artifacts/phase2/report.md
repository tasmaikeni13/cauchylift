# Phase 2 stochastic, boundary, scale, and full-model theory

Run ID: `phase2-20260828T103838Z`

Audit date: 2026-08-28 UTC

Required Phase 1 handoff: `5221b458405119b0da297500b210ddc7a34bf0c5` (`PASS`, verified on `origin/main`)

Scope: mathematical analysis, deterministic CPU diagnostics, machine-readable specification, and Lean; no training stack and no GPU workload

## Decision summary

Phase 2 passes without replacing the primitive. The exact update is now total and frozen for scalars, vectors, matrices, embeddings/heads, normalization gains, and higher-order tensors. The finite-precision execution rule uses projectively neutral max scaling, nonnegative exclusion sums, FP32-or-higher accumulation, and an FP64 rare path; no additive epsilon is permitted.

The stochastic result is conditional rather than universal. It gives positive expected alignment and smooth one-step descent under directly measurable noise margins. An unbiased scalar distribution with expected ascent remains a mandatory counterexample. The old informal “mode deflation” story is replaced by an exact local signature: on an isolated diagonal two-mode quadratic, exact line search maps (q\) to \(-q^{-3}\), but two idealized steps map it to \(q^9\). That is a sharp mode-alternation prediction and an equally sharp concentration risk, not an acceleration theorem.

## Requirement-level results

### Normalized simplex and boundary strata

For (X=G/\|G\|_F\), energy shares (a_{ij}=X_{ij}^2\), row/column masses (R_i,C_j\), and normalized denominators (h_{ij}=2-R_i-C_j\), the projective direction is the normalization of (X_{ij}/h_{ij}\).

If a dominant cell has share (1-\tau\), its denominator lies in ([\tau,2\tau]\), while every other denominator is at least (1-\tau\). This gives the proved one-sparse modulus

\[
\left\|\operatorname{CL}(G)/\rho-e_p\right\|_F
\le2\left(\tau/(1-\tau)\right)^{3/2}.
\]

The active denominator ratio is at most (2/\tau\) and grows at order (1/\tau\) in the recorded cases. On regions with (h_{ij}\ge\delta>0\), the unit-radius map is Lipschitz with constant (4/\delta+16/\delta^2\). The angle guarantee has no equality case: the cosine is strictly greater than (1/\sqrt3\) for every nonzero input. No continuous extension exists at zero because distinct scale-invariant rays remain separated as both inputs vanish.

### Stochastic gradients

Let (\mu=\nabla f(W)\), let (g\) be unbiased, (D=\operatorname{CL}(g)\), (\gamma=1/\sqrt3\), and (\rho=\sqrt{\min(m,n)}\). The pointwise inequality

\[
\langle\mu,D\rangle\ge\rho(\gamma\|g\|_F-\|g-\mu\|_F)
\]

implies

\[
\mathbb E\langle\mu,D\rangle
\ge\rho(\gamma\|\mu\|_F-\sigma),
\qquad
\sigma^2=\mathbb E\|g-\mu\|_F^2.
\]

Thus (\sigma<\gamma\|\mu\|_F\) is sufficient for positive expected alignment. The paper also gives a first-moment form, a high-probability ((\kappa,\zeta)\) form, an interior transformation-bias bound, and the corresponding (L\)-smooth one-step descent interval. All assumptions map to estimable large-batch/microbatch quantities.

The negative distribution (g=10\) with probability (0.1\), (g=-0.5\) with probability (0.9\) has mean (0.55\) but expected CauchyLift direction (-0.8\), hence alignment (-0.44\). Unbiasedness alone is formally rejected.

### Mechanism

For an active diagonal gradient ratio (q=g_1/g_2\), CauchyLift's direction ratio is (q^3\). Exact line-search orthogonality gives (q^+=-q^{-3}\), independent of the two positive curvature values. Normalized gradient gives exponent (-1\); sign, fully row/column-normalized, and polar controls give unit next-ratio magnitude. The diagnostic signature is therefore a local log-ratio slope of (-3\).

Applying the idealized rule twice gives (q^{++}=q^9\). This retained result prevents any claim of monotone balancing or unconditional acceleration. Phase 3 must measure alternating concentration and reject the mechanism where the local two-mode model should apply but the slope is absent.

### Width, shape, and full decoder semantics

A fixed radius (\rho\) has average squared row and column norms (\rho^2/m\) and (\rho^2/n\). Requiring both to be at most one yields (\rho^2\le\min(m,n)\); the maximal transpose-symmetric choice is the frozen (\rho=\sqrt{\min(m,n)}\).

The initial model contract is a bias-free decoder with trainable normalization gains. Stored matrices keep their shapes; scalars use (1\times1\); vectors use (d\times1\); higher tensors flatten the semantic first/output axis against all remaining axes; tied embedding/head weights are transformed once after shared-gradient accumulation. Every parameter uses CauchyLift. Sparse layouts must be mathematically equivalent, with no Adam/SGD fallback.

### Finite precision

The old reference path used (2S-r-c\) subtraction and can falsely produce a zero complement near a dominant cell. It was repaired to use linear-work prefix/suffix exclusion sums. Max-absolute scaling keeps every square at most one; multiplying by the minimum positive active denominator keeps the raw representative bounded before final normalization. FP32 is the minimum accumulator, and FP64 recomputes any active denominator that still rounds to zero. Exactly represented one-sparse inputs take the declared boundary branch.

A fixed additive (10^{-3}S\) epsilon changes an ordinary direction by approximately (6.1\times10^{-4}\) in the recorded case. It is therefore a different optimizer and is forbidden. BF16 and FP16 represented cases match the decimal oracle in the deterministic suite.

## Expanded verification

- 1,554 exhaustively enumerated nonzero matrices over ({-1,0,1}\) in the declared small shapes;
- 10,000 additional deterministic random property cases over shapes through (6\times6\);
- boundary modulus, denominator range, regional Lipschitz, and zero-discontinuity checks;
- exact finite-distribution stochastic bounds, bias, smooth descent, and expected ascent;
- exact two-mode recurrences against four controls;
- width families from 64 through 1,024 plus representative decoder parameter shapes;
- BF16, FP16, decimal-oracle, overflow, underflow, cancellation, and epsilon cases;
- every pre-existing deterministic analysis, including all adverse quadratic/rank/rejection results;
- the full Lean target, extended with the deterministic noise-alignment core and exact two-mode identities.

No new result invalidated the primitive itself, so the theory-repair replacement loop was not triggered. The expected-ascent, (q^9\), sparse-concentration, rotated-quadratic, and stable-rank negatives are instead promoted to implementation/experiment kill diagnostics.

## Gate result

**PASS** — the exact update is total and numerically specified for every intended trainable shape; stochastic use has valid descent results under explicit measurable assumptions plus a precise counterprediction; the radius is derived and frozen; the mechanism has a theorem-level, control-distinguishing signature beyond algebraic rank; the expanded deterministic and Lean suites pass with all negative cases visible; the paper separates proved, machine-checked, numerically checked, hypothesized, and unknown statements; and `spec/optimizer_v0.2.json` is the authoritative Phase 3 handoff.
