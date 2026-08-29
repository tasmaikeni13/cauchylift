# CauchyLift

CauchyLift is a theory-stage optimizer proposal built around one new primitive: a **cotransverse rational gradient field**. For a matrix gradient \(G\), it measures the gradient energy outside each entry's row and outside its column, divides that entry by the combined excluded energy, and normalizes the resulting field once.

This repository intentionally contains no model-training implementation and makes no neural-training performance claim. It contains the research paper, a dated novelty audit, formal Lean statements, deterministic mathematical analyses, and the Phase 3 PyTorch/native-HIP optimizer implementation. The design target is matrix-aware convergence behavior with only linear work, reductions, pointwise operations, and no persistent optimizer state. Whether that target survives neural-network experiments is an open, explicitly falsifiable question.

## Core map

For \(G\in\mathbb R^{m\times n}\setminus\{0\}\), let

\[
S=\lVert G\rVert_F^2,\qquad
r_i=\sum_jG_{ij}^2,\qquad
c_j=\sum_iG_{ij}^2,
\]

and define the cotransverse energy

\[
E_{ij}=(S-r_i)+(S-c_j)=2S-r_i-c_j.
\]

Away from the one-sparse boundary, set \(Z_{ij}=G_{ij}/E_{ij}\) and

\[
\operatorname{CL}(G)=
\sqrt{\min(m,n)}\;\frac{Z}{\lVert Z\rVert_F}.
\]

At the one-sparse boundary, the map is the projective limit of
\(G_{ij}/(E_{ij}+\varepsilon S)\) as \(\varepsilon\downarrow0\). This is a definition of the primitive, not a tunable stabilizer.

The update is \(W_{t+1}=W_t-\eta_t\operatorname{CL}(\nabla f(W_t))\). Momentum, running moments, matrix roots, rotations, whitening, clipping, and weight decay are not part of the proposed primitive.

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

- No language-model or neural-network training has been run.
- Algebraic rank lift is not the same as useful numerical stable rank; the included probe demonstrates this distinction.
- The finite literature search supports only a scoped statement that no close formula was found through 2026-08-28. It cannot prove that nobody has ever considered an equivalent map.

## Repository map

- [phases/](phases/) — eight gated prompts covering theory repair, ROCm implementation, controlled scaling, the 125M/1B-token study, and a submission-ready paper.
- [`paper/paper.md`](paper/paper.md) — full paper with derivations, theorems, limitations, and proposed empirical protocol.
- [`research/`](research/) — research contract, query ledger, closest-work matrix, rejected candidates, claim audit, and risk register.
- [`analysis/`](analysis/) — standard-library-only mathematical and numerical probes; no training code.
- [`formal/`](formal/) — Lean project and proof audit.
- [`cauchylift/`](cauchylift/) and [`csrc/`](csrc/) — reference optimizer and native HIP kernels.
- [`docs/phase3.md`](docs/phase3.md) — isolated environment, build, tests, benchmark, profiler, and declared tolerances.

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

## Status

**Theory-stage research hypothesis, version 0.2.0.** Phases 1 and 2 passed on
2026-08-28, and Phase 3 passed on 2026-08-29. The original two-GPU-family
criterion remains unreplicated; the Phase 3 result is a single-MI300X result.
Phase 4 has not been started. No neural-network training has been run.
