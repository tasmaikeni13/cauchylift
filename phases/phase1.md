# Phase 1 prompt — adversarial novelty and mathematical audit

Work autonomously in the CauchyLift repository and complete Phase 1. Read phases/README.md first and obey its state machine. This is a theory and novelty phase: do not create a neural-network training loop and do not spend meaningful GPU time.

## Objective

Try to disprove the present CauchyLift proposal before strengthening it. Determine whether the cotransverse rational field is actually a new, non-compositional mathematical primitive, whether its current statements are correct, and whether the claimed mechanism is coherent enough to justify implementation. The desired outcome is not a favorable review; it is a defensible object that survives hostile scrutiny.

## Required work

1. Audit the exact map, projective boundary convention, scale invariance, oddness, transpose symmetry, fixed-radius normalization, descent-angle theorem, smooth deterministic convergence theorem, Cauchy factorization, and algebraic rank-lift result. Re-derive each statement independently. Search for missing hypotheses, undefined shapes, zero-gradient behavior, numerical singularities, equality cases, and proof steps that do not survive edge cases.
2. Construct adversarial examples symbolically and numerically. Include one-sparse and nearly one-sparse matrices, one-row and one-column tensors, scalars, extreme dynamic range, low precision, repeated row or column energies, rank-deficient inputs, anisotropic quadratics, rotated quadratics, and stochastic sign reversals.
3. Refresh the novelty search through the current date. Search primary papers, proceedings, preprints, theses, patents when searchable, and public optimizer implementations. Use formula fragments and mathematical concepts, not only the name CauchyLift. Compare exact defining equations against marginal normalization, complement statistics, leave-one-row/column methods, Cauchy kernels, rational gradient fields, graph incidence or cut energies, mirror maps, norm geometry, matrix scaling, whitening, Muon/polar methods, SOAP/Shampoo, SinkGD, Scion, SWAN, Adafactor, and recent stateless optimizers.
4. Update research/search_log.md, research/closest_work_matrix.md, research/evidence_ledger.md, and research/risk_register.md with dated primary-source evidence. Distinguish formula equivalence, conceptual proximity, and superficial vocabulary overlap. Never write that the whole ML community has never seen the idea; state the searchable scope and remaining uncertainty.
5. Check whether the rule can be rewritten as a known optimizer plus a module, a familiar normalization fixed point, a standard mirror/steepest-descent step, or a renamed preconditioner family. A generic representation as P(G)G is not by itself disqualifying; an operational decomposition into known reusable modules is.
6. If any theorem or definition is false, repair it immediately in paper/paper.md, research/proof_audit.md, the analysis scripts, and formal/CauchyLift.lean where applicable. Add regression cases. Do not weaken a claim silently.
7. If an exact or substantively equivalent prior primitive is found, or if the core construction is mathematically unusable, mark the current candidate FAIL_CORE. Derive and screen at least three first-principles replacement primitives that satisfy the hard exclusions. Record all in research/candidate_portfolio.md, with explicit collision tests and counterexamples. Promote none without the same hostile audit.
8. Run every existing deterministic analysis and the full Lean build. Make results reproducible and update the paper wherever the audit changes a claim.

## Gate

Phase 1 passes only if:

- every active theorem has a correct written proof and all declared Lean targets build;
- edge-case semantics cover every tensor shape intended for later training, or the missing coverage is explicitly deferred to Phase 2 with no implementation ambiguity;
- no searched source contains the same defining primitive or an operationally equivalent composition;
- novelty language is scoped, dated, and supported by a query ledger;
- all counterexamples and negative results remain visible;
- the primitive still meets every hard exclusion in research/research_contract.md.

Write the standard Phase 1 artifacts. End the report with one of PASS, REVISE, FAIL_CORE, or BLOCKED and exact evidence. Commit and push validated work without force. Do not start Phase 2 in this session.
