# Risk register

| Risk | Severity | Evidence now | Required resolution |
|---|---:|---|---|
| Absolute novelty is unknowable from finite search | High | No exact collision found in the dated search | Use scoped language; repeat search before release claims |
| Complement denominator concentrates the update too sharply | High | Condition-10,000 scheduled quadratics underperform sign/Sinkhorn in some settings | Measure update concentration, loss spikes, and tune fairly in later training |
| Algebraic rank is numerically meaningless | High | Stable rank remains approximately 1 in sampled rank-one probes | Do not claim spectral diversity; examine numerical spectrum only as a diagnostic |
| Exact-gradient theorem may not survive stochastic bias | High | Nonlinear map does not preserve unbiasedness | Prove under a transparent oracle-alignment condition or derive a genuine noise theorem |
| Reduction kernels may be bandwidth/launch bound | Medium | Complexity analysis only | Build a fused kernel and benchmark optimizer step time later |
| Width scaling \(\sqrt{\min(m,n)}\) may be wrong | Medium | Chosen to match partial-isometry Frobenius radius, not derived for CauchyLift | Run width-transfer study and derive architecture-aware perturbation bounds |
| Basis dependence may harm rotated problems | High | At condition 10,000 all exact-line rotated CauchyLift runs hit the cap | Treat as a primary kill test; do not add a rotation wrapper |
| One-sparse projective boundary may be numerically abrupt | Medium | Exact limit is defined and property tests pass | Measure near-boundary continuity in finite precision and derive an implementation-safe equivalent |
| Lack of momentum may hurt noisy language modeling | High | No training performed | Test the primitive alone first; needing momentum is a contract-level negative result |
| Weight decay omitted | Low | Deliberate scope choice | Evaluate regularization separately; do not fold it into the novelty claim |
