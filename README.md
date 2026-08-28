# CauchyLift

CauchyLift is a theory-stage optimizer proposal built around one new primitive: a **cotransverse rational gradient field**. For a matrix gradient \(G\), it measures the gradient energy outside each entry's row and outside its column, divides that entry by the combined excluded energy, and normalizes the resulting field once.

This repository intentionally contains no model-training implementation and makes no empirical performance claim. It contains the research paper, a dated novelty audit, formal Lean statements, and deterministic mathematical analyses. The design target is matrix-aware convergence behavior with only linear work, reductions, pointwise operations, and no persistent optimizer state. Whether that target survives neural-network experiments is an open, explicitly falsifiable question.

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

## What is not established

- No language-model or neural-network training has been run.
- No wall-clock comparison with AdamW, Muon, SOAP, or any other optimizer has been run.
- Algebraic rank lift is not the same as useful numerical stable rank; the included probe demonstrates this distinction.
- The finite literature search supports only a scoped statement that no close formula was found through 2026-08-28. It cannot prove that nobody has ever considered an equivalent map.

## Repository map

- [phases/](phases/) — eight gated prompts covering theory repair, ROCm implementation, controlled scaling, the 125M/1B-token study, and a submission-ready paper.
- [`paper/paper.md`](paper/paper.md) — full paper with derivations, theorems, limitations, and proposed empirical protocol.
- [`research/`](research/) — research contract, query ledger, closest-work matrix, rejected candidates, claim audit, and risk register.
- [`analysis/`](analysis/) — standard-library-only mathematical and numerical probes; no training code.
- [`formal/`](formal/) — Lean project and proof audit.

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

## Status

**Theory-stage research hypothesis, version 0.2.0.** Phases 1 and 2 passed on 2026-08-28. Boundary, stochastic, mechanism, shape/radius, and finite-precision semantics are frozen in [`spec/optimizer_v0.2.json`](spec/optimizer_v0.2.json). Kernel prototyping and controlled training experiments remain deliberately absent here.
