# Risk register

| Risk | Severity | Evidence now | Required resolution |
|---|---:|---|---|
| Absolute novelty is unknowable from finite search | High | No exact collision found in the dated search | Use scoped language; repeat search before release claims |
| Complement denominator concentrates the update too sharply | High | Condition-10,000 scheduled quadratics underperform sign/Sinkhorn in some settings | Measure update concentration, loss spikes, and tune fairly in later training |
| Algebraic rank is numerically meaningless | High | Stable rank remains approximately 1 in sampled rank-one probes | Do not claim spectral diversity; examine numerical spectrum only as a diagnostic |
| Conditional stochastic margin may fail in training | High | Phase 2 proves descent when \(\sigma<\|\mu\|/\sqrt3\), but the retained unbiased scalar distribution has expected ascent | Measure minibatch SNR/alignment; do not claim unconditional stochastic convergence |
| Reduction kernels may be bandwidth/launch bound | Medium | Complexity analysis only | Build a fused kernel and benchmark optimizer step time later |
| Frozen width scaling may be empirically wrong | Medium | Phase 2 derives \(\sqrt{\min(m,n)}\) as the maximal symmetric average-fiber-capped radius and width identities pass | Keep it frozen initially; diagnose transfer without tuning on held-out results |
| Basis dependence may harm rotated problems | High | At condition 10,000 all exact-line rotated CauchyLift runs hit the cap | Treat as a primary kill test; do not add a rotation wrapper |
| Boundary kernel may violate the stable projective rule | Medium | Phase 2 proves an \(O(\tau^{3/2})\) modulus and finds subtractive cancellation in the old reference path | Use exclusion-safe sums, FP32 accumulation, FP64 rare path, and no epsilon; validate kernels against oracle |
| Lack of momentum may hurt noisy language modeling | High | No training performed | Test the primitive alone first; needing momentum is a contract-level negative result |
| Weight decay omitted | Low | Deliberate scope choice | Evaluate regularization separately; do not fold it into the novelty claim |
| Generic rank hypotheses are easy to understate | Medium | Phase 1 found that distinct nonzero magnitudes still allow zero factor coordinates and rank loss | Keep the all-entries-nonzero condition in the paper, proof audit, and regression suite |
| Mode alternation can become harmful concentration | High | Exact two-mode law gives \(q^+=-q^{-3}\) and \(q^{++}=q^9\) | Measure local log-slope, update concentration, and loss spikes; do not market the law as acceleration |
| Sparse embedding gradients can concentrate the fixed radius | High | Shape semantics are total, but no sparse training diagnostic exists | Count active rows and per-row update norms; no fallback optimizer is allowed |
| Interior sensitivity bound is too loose for engineering thresholds | Medium | Proved constant is much larger than observed local ratios | Use it only as a correctness theorem; obtain measured kernel/error thresholds in Phase 3 |
| FP64 denominator rare path could be frequent or slow | Medium | Defined by the machine spec but not measured | Instrument branch count and benchmark it on adversarial and representative gradients |
