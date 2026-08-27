# Phase 2 prompt — stochastic, boundary, scale, and full-model theory

Work autonomously in the CauchyLift repository and complete Phase 2. Read phases/README.md and require a PASS handoff from Phase 1. This remains a mathematical phase; GPU use is limited to small diagnostics.

## Objective

Close the gap between the present exact-gradient matrix theory and a real decoder-only Transformer. Develop genuinely informative mathematics for stochastic gradients, boundary stability, width scaling, and all trainable parameter shapes. If the mechanism fails, repair or replace the theory before any training system is built.

## Required work

1. Express the cotransverse field on the normalized energy simplex and study its boundary strata. Characterize continuity of the projective extension, equality cases of the angle bound, sensitivity to perturbations, denominator dynamic range, and any Lipschitz or Hölder region that can be proved. Find sharp counterexamples outside proved regions.
2. Analyze the nonlinear stochastic update. The deterministic angle theorem for CL(grad f) does not automatically apply to CL(g) for a noisy minibatch gradient g. Derive useful sufficient conditions for positive expected descent alignment, bias and variance controls, or a high-probability bound. Test necessity with explicit adversarial noise distributions. If only conditional results are possible, state the assumptions in measurable training quantities.
3. Investigate the proposed mode-deflation mechanism on quadratic and local linearized dynamics. Derive predictions that distinguish CauchyLift from normalized gradient descent, sign descent, row/column normalization, SinkGD, and polar or whitening methods. A rank increase alone is not an acceleration explanation. Produce at least one theorem or rigorously bounded proposition with a falsifiable empirical signature, or remove the mechanism claim.
4. Resolve width and shape semantics. Derive the update-radius scaling for rectangular matrices, embeddings, output heads, one-row or one-column parameters, and scalars. Ensure every trainable Transformer parameter uses one coherent primitive, not an Adam/SGD fallback. Prefer a bias-free architecture or a mathematically native treatment over hidden composition. State parameter grouping and zero-gradient behavior exactly.
5. Analyze finite precision. Determine safe accumulation precision, overflow/underflow regimes, boundary detection, and whether any epsilon changes the mathematical map. Fixed numerical safeguards must implement the declared projective rule, not become tunable optimizer components.
6. Extend the deterministic analysis suite with property-based, exhaustive small-shape, stochastic-noise, width-transfer, and finite-precision tests. Add symbolic or high-precision oracles where useful. Keep test cases deterministic and record seeds.
7. Formalize the most central new result in Lean when feasible. If a full stochastic theorem is not practical, formalize the exact deterministic lemma on which it depends and state the remaining probabilistic proof boundary precisely.
8. Rewrite the relevant definition, theory, mechanism, limitations, and protocol sections of paper/paper.md. Update the proof audit, research contract, risk register, and evidence ledger. Do not retain prose that the new analysis falsifies.

## Theory-repair loop

If a counterexample invalidates the primitive rather than a peripheral theorem, stop downstream progress. Reopen the Phase 1 collision search, generate multiple non-compositional replacements, and screen them against the existing rejection ledger. Any replacement must have a clean mathematical definition, linear-work implementation model, descent evidence, full-shape semantics, and a new dated novelty audit. Version the method and mark all older downstream evidence invalid.

## Gate

Phase 2 passes only if:

- the exact update is total and numerically implementable for every intended trainable parameter shape;
- stochastic use has either a valid descent result under explicit measurable assumptions or a precise counterprediction and kill test;
- the width/radius choice is derived and frozen for initial experiments;
- the claimed mechanism has a theorem-level or sharply testable basis beyond algebraic rank;
- the expanded deterministic suite passes, the formal artifact builds, and all negative cases are reported;
- the paper cleanly separates proved, numerically checked, hypothesized, and unknown statements.

Write the standard Phase 2 artifacts, including a machine-readable specification of the optimizer for Phase 3. End with the exact gate result. Commit and push validated work without force. Do not implement the training stack in this session.
