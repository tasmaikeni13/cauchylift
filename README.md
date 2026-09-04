# CauchyLift

CauchyLift is an optimizer primitive built around an **additive fiber root-mean-square (RMS) Cauchy kernel**. For a matrix gradient \(G\in\mathbb R^{m\times n}\), it measures the intrinsic RMS energy of each coordinate's row fiber and column fiber, scales the entry by the sum of fiber RMS energies (forming an exact additive Cauchy kernel), and normalizes the resulting field to the longest dimension radius.

The design target is matrix-aware 2nd-order-like Riemannian convergence behavior with strictly linear work \(O(mn)\), pointwise and reduction operations only, and strictly **zero persistent optimizer state**.

## Core map (CauchyLift v0.3)

For \(G\in\mathbb R^{m\times n}\setminus\{0\}\), let

\[
r_i=\sum_{j=1}^n G_{ij}^2,\qquad
c_j=\sum_{i=1}^m G_{ij}^2,
\]

and define the fiber RMS denominator

\[
D_{ij} = \text{RMS}(G_{i,:}) + \text{RMS}(G_{:,j}) = \sqrt{\frac{r_i}{n}} + \sqrt{\frac{c_j}{m}}.
\]

For rank-one gradients \(G = a b^T\), \(D_{ij} = |a_i|\text{RMS}(b) + |b_j|\text{RMS}(a)\), forming an exact additive Cauchy matrix \(C_{ij} = \frac{1}{x_i + y_j}\).

On active entries (\(D_{ij} > 0\)), set the raw field \(Z_{ij}=G_{ij}/D_{ij}\) and define the update direction:

\[
\operatorname{CL}(G)=
\sqrt{\max(m,n)}\;\frac{Z}{\lVert Z\rVert_F}.
\]

At the one-sparse boundary, the map contracts to \(\operatorname{sgn}(G_{ij})\sqrt{\max(m,n)}\). The update is \(W_{t+1}=W_t-\eta_t\operatorname{CL}(\nabla f(W_t))\). Momentum, running moments, matrix roots, rotations, whitening, clipping, and weight decay are strictly not part of the proposed primitive.

## What is established here

- Linear arithmetic work \(O(mn)\), row/column reductions, pointwise arithmetic, and no persistent optimizer state.
- Scale invariance, oddness, fixed Frobenius update norm, sign preservation, and strict descent alignment for every nonzero exact gradient.
- A dimension-independent angle guarantee
  \(\langle G,\operatorname{CL}(G)\rangle\ge
  \lVert G\rVert_F\lVert\operatorname{CL}(G)\rVert_F/\sqrt3\).
- A deterministic \(O(T^{-1/2})\) stationarity bound for smooth objectives under normalized steps.
- A continuous one-sparse projective extension with an explicit boundary modulus and an interior Lipschitz bound.
- Conditional expected stochastic descent under measurable minibatch-noise assumptions, plus an explicit unbiased-noise expected-ascent counterexample.
- Frozen scalar, vector, matrix, embedding, and higher-tensor semantics with a machine-readable finite-precision specification.
- An exact two-mode mode-alternation signature \(q^+=-q^{-3}\), reported with its harmful \(q^{++}=q^9\) counterprediction.
- A Cauchy-kernel factorization on rank-one gradients and a generic algebraic rank-lift theorem.
- Reproducible property checks and synthetic quadratic probes, including negative results that prevent premature performance claims.
- FP64-oracle, PyTorch, and native HIP numerical agreement on the single MI300X with zero persistent optimizer state.
- A five-kernel multi-tensor HIP step whose representative BF16 median is
  1.0625 ms versus 0.9602 ms for fused AdamW (1.1065×), passing the Phase 3
  1.15× engineering gate on the inventoried MI300X.

## What is not established

- Large-scale pretraining (125M / 1B tokens and 350M / 3B tokens on 8x MI300X) is preregistered in Phases 6–8 and remains to be executed.
- The finite literature search supports only a scoped statement that no close formula was found through 2026-08-28. It cannot prove that nobody has ever considered an equivalent map.

## Repository map

- [phases/](phases/) — nine gated prompts covering theory repair, ROCm implementation, controlled scaling pilots, 125M / 1B-token pretraining, 350M / 3B-token flagship pretraining on 8x MI300X, and a submission-ready paper.
- [`paper/paper.md`](paper/paper.md) — full paper with derivations, theorems, limitations, and proposed empirical protocol.
- [`research/`](research/) — research contract, query ledger, closest-work matrix, rejected candidates, claim audit, and risk register.
- [`analysis/`](analysis/) — standard-library-only mathematical and numerical probes; no training code.
- [`formal/`](formal/) — Lean project and proof audit.
- [`cauchylift/`](cauchylift/) and [`csrc/`](csrc/) — reference optimizer and native HIP kernels.
- [`docs/phase3.md`](docs/phase3.md) — isolated environment, build, tests, benchmark, profiler, and declared tolerances for Phase 3.
- [`docs/phase4.md`](docs/phase4.md) — neutral training system, decoder-only Transformer, verified ROCm FlashAttention, baseline suite, and data pipeline for Phase 4.

## Reproduce the mathematical analyses

Python 3.11 or newer is sufficient; there are no third-party Python dependencies.

```bash
python3 analysis/run_property_checks.py
python3 analysis/run_quadratic_suite.py
python3 analysis/run_rank_probe.py
python3 analysis/run_rejection_checks.py
python3 analysis/run_adversarial_audit.py
python3 analysis/run_boundary_suite.py
python3 analysis/run_stochastic_suite.py
python3 analysis/run_mechanism_suite.py
python3 analysis/run_width_suite.py
python3 analysis/run_finite_precision_suite.py
```

Each command prints deterministic JSON. The checked-in outputs under `analysis/results/` were generated with those commands.

For Lean, install Lean via `elan`, then run:

```bash
cd formal
lake update
lake build
```

See [`formal/README.md`](formal/README.md) for the exact proof boundary.

## Reproduce Phase 3

The ROCm environment, CPU/GPU tests, benchmark, and profiler commands are in
[`docs/phase3.md`](docs/phase3.md). CPU-only continuous tests skip ROCm cases
honestly; no model or dataset is built.

## Reproduce Phase 4

The neutral training system, test suite (90 tests), overfit smoke tests across all 7 optimizers, deterministic checkpoint resumption, and profiler traces are documented in
[`docs/phase4.md`](docs/phase4.md). Run:

```bash
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build .venv/bin/python -m pytest -q
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build .venv/bin/python scripts/run_phase4_smoke.py
```

## Status

**Research prototype, version 0.3.0.** Phases 1 and 2 passed on
2026-08-28, Phase 3 passed on 2026-08-29, and Phase 4 passed on 2026-09-04.
The original two-GPU-family criterion remains unreplicated; results are on the single MI300X.
Phase 5 (small-scale falsification and baseline screen) completed on 2026-09-04 with gate `PASS`.
Following the theory-repair loop, CauchyLift v0.3 implemented the additive fiber RMS Cauchy kernel $D_{ij} = \text{RMS}(G_{i,:}) + \text{RMS}(G_{:,j})$ (formally verified in Lean 4) and longest-fiber radius scaling $\rho = \sqrt{\max(m, n)}$. Under a strict 170-run equal-budget screen across 4 workloads (Small LM, Medium LM, Small ViT, Held-Out ConvSSM), CauchyLift v0.3 beat tuned AdamW on 3 of the 4 workloads (W1: 7.0413 vs 7.1080; W2: 7.0991 vs 7.1427; W3: 1.9107 vs 1.9226) with fused kernel step time of 0.29–0.41 ms and strictly zero persistent optimizer state. Full details in [`artifacts/phase5/report.md`](artifacts/phase5/report.md) and [`docs/phase5.md`](docs/phase5.md).
Phase 6 is authorized to execute the multi-GPU scaling pilot and freeze dual preregistrations for both 125M (1B tokens) and 350M (3B tokens) on FineWeb-Edu across 8x AMD Instinct MI300X.
