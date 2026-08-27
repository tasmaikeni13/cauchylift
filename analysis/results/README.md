# Checked-in numerical results

These JSON files are deterministic mathematical probes, not neural-network experiments.

- `property_checks.json` — 5,000 implementation/property cases; all declared checks pass, minimum observed gradient-direction cosine 0.924220.
- `quadratic_suite.json` — held-out 4×4 analytic Kronecker quadratics at condition numbers \(10^2\) and \(10^4\), with both exact line search and a tuned inverse-square-root schedule.
- `rank_probe.json` — exact rational rank lift plus a floating stable-rank countercheck.
- `rejection_checks.json` — evidence used to reject the polar-equivalent exterior branch and unstable plaquette-dual branch.

The hard-condition quadratic results and stable-rank probe are intentionally adverse. They prevent the paper from converting attractive algebra into an unsupported performance claim.

Regeneration commands and environment versions are in [`REPRODUCIBILITY.md`](../../REPRODUCIBILITY.md).
