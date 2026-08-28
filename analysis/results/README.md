# Checked-in numerical results

These JSON files are deterministic mathematical probes, not neural-network experiments.

- `property_checks.json` — 5,000 implementation/property cases; all declared checks pass, minimum observed gradient-direction cosine 0.924220.
- `quadratic_suite.json` — held-out 4×4 analytic Kronecker quadratics at condition numbers \(10^2\) and \(10^4\), with both exact line search and a tuned inverse-square-root schedule.
- `rank_probe.json` — exact rational rank lift plus a floating stable-rank countercheck.
- `rejection_checks.json` — evidence used to reject the polar-equivalent exterior branch and unstable plaquette-dual branch.
- `adversarial_audit.json` — Phase 1 boundary/shape/dynamic-range audit with retained rank-hypothesis and unbiased-noise counterexamples.
- `boundary_suite.json` — exhaustive small shapes, 10,000 seeded properties, boundary modulus, denominator range, interior sensitivity, and zero-discontinuity checks.
- `stochastic_suite.json` — exact finite-distribution alignment, bias, smooth-descent, and expected-ascent cases.
- `mechanism_suite.json` — cubic two-mode exact-line recurrence and control-method counterpredictions.
- `width_suite.json` — frozen radius identities and total decoder parameter-shape semantics.
- `finite_precision_suite.json` — BF16/FP16, decimal-oracle, cancellation, overflow, underflow, and epsilon-collision diagnostics.

The hard-condition quadratic results and stable-rank probe are intentionally adverse. They prevent the paper from converting attractive algebra into an unsupported performance claim.

Regeneration commands and environment versions are in [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).
