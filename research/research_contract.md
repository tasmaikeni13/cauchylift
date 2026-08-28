# Research contract

Originally frozen on 2026-08-27 UTC; Phase 2 implementation semantics frozen on 2026-08-28 UTC.

## Objective

Develop a genuinely non-compositional optimizer hypothesis for matrix-shaped neural-network parameters with:

1. a primitive not obtained by appending momentum, Adam moments, rotation, whitening, low-rank projection, or a scheduler to an existing optimizer;
2. linear arithmetic work in the number of parameters, GPU-regular reductions and pointwise operations, and no matrix decomposition or matrix-matrix iteration;
3. a credible mathematical mechanism that could eventually compete with matrix optimizers in convergence per step;
4. explicit theorems, machine-checked proof fragments, and reproducible numerical analysis before any training code exists.

## Hard exclusions

A candidate is rejected if its defining transform is any of the following:

- Adam/AdamW, Lion, SGD, or sign descent plus another module;
- a cheap or blockwise approximation to a polar factor, whitening, Shampoo, SOAP, or Muon;
- a rotation followed by a known elementwise optimizer;
- a row/column normalization fixed point, Sinkhorn iteration, or ordinary matrix scaling;
- a low-rank subspace method, spectral truncation, or covariance/Fisher approximation;
- a norm-ball linear minimization oracle with a newly chosen norm;
- a known temporal extrapolator rewritten in geometric language;
- a method whose safety depends on blending its direction with a conventional optimizer.

The fact that any descent direction can be represented post hoc as \(P(G)G\) for some positive semidefinite, gradient-dependent operator makes “not a preconditioner” an impossible criterion. The operational criterion is instead that the rule is derived as one closed primitive and does not consist of reusable optimizer modules placed in series or parallel.

## Frozen primitive

The surviving object is the cotransverse rational field

\[
G_{ij}\longmapsto
\frac{G_{ij}}{(\|G\|_F^2-\|G_{i,:}\|_2^2)+(\|G\|_F^2-\|G_{:,j}\|_2^2)},
\]

followed only by the scalar normalization required to assign an update radius. Its denominator is the sum of energy outside the entry's row and outside its column. This **complement**, rather than a row or column statistic itself, is the defining departure from existing marginal-normalization families.

## Phase 2 frozen semantics

- Every trainable parameter uses the same primitive. Scalars map to \(1\times1\), vectors to \(d\times1\), matrices keep their stored shape, and higher-order tensors flatten the first semantic output/record axis against the remaining axes.
- The update radius is \(\rho_{m,n}=\sqrt{\min(m,n)}\), the maximal transpose-symmetric radius whose average squared row and column update norms are both at most one. It is frozen for initial experiments rather than tuned by layer.
- Zero gradients map to zero. Exactly one-sparse represented gradients use the signed projective boundary. All other active denominators are positive mathematically and must be computed with exclusion-safe nonnegative sums.
- Accumulation is FP32 or higher, with a rare FP64 complement recomputation when an active FP32 denominator rounds to zero. No additive epsilon, momentum, moment, clipping, fallback, or learned stabilizer is permitted.
- The initial decoder is bias-free with trainable normalization gains. Embeddings, tied or untied heads, attention/MLP matrices, gain vectors, and scalars all remain inside this contract.

The authoritative machine-readable form is `spec/optimizer_v0.2.json`.

## Claim levels

- **Proved:** exact algebraic statements supported by a written proof and, where listed, Lean.
- **Numerically checked:** deterministic scripts with fixed seeds; never generalized to neural training.
- **Hypothesized:** plausible mechanism with a stated counterprediction and kill threshold.
- **Not claimed:** wall-clock speed, language-model convergence, generalization, stochastic convergence without an alignment assumption, or absolute historical novelty.

## Primary predictions

1. The field remains uniformly descent aligned despite its reciprocal complement coupling.
2. On an isolated diagonal two-mode local model, the exact-line gradient ratio follows \(q^+=-q^{-3}\). Later diagnostics should recover this mode-alternation signature where the approximation is valid; the signature does not itself imply acceleration.
3. The transform can change algebraic rank through a Cauchy kernel even when the input is rank one.
4. A fused implementation should require only row/column reductions and elementwise passes, with no persistent state.

## Counterpredictions and kill criteria

The research hypothesis should be killed or materially redesigned if any of these occur in the later empirical phase:

- tuned CauchyLift does not beat tuned AdamW in tokens-to-target on at least three of four predeclared workloads;
- its optimizer-step wall time exceeds AdamW by more than 15% in a fused implementation on two modern GPU families;
- the denominator field repeatedly causes update concentration, loss spikes, or a lower stable rank than the raw gradient without a task-level gain;
- performance requires momentum, Adam moments, clipping, rotation, or a baseline-direction mixture to become competitive;
- a prior source is found with the same cotransverse denominator and normalized rational update;
- gains disappear under equal tuning budgets or reverse on the held-out workloads.

## Deliverable boundary

This repository stops at theory and mathematical analysis. It includes no optimizer integration, CUDA kernel, model, dataset, or training loop.
