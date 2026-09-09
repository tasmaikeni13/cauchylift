# Phase 2 Prompt — Stochastic Dynamics, Fiber Curvature Bounds, and Scaling Theory

Work autonomously in the CauchyLift repository and complete Phase 2. Read `phases/README.md` and require a PASS handoff from Phase 1. This remains a theoretical and diagnostic phase; GPU use is limited to small micro-benchmarks.

## Objective

Close the gap between exact matrix theory and deep neural network dynamics. Analyze the interaction between stochastic minibatch noise and the momentum-filtered velocity manifold, establish fiber curvature adaptation bounds, derive width/radius scaling for all Transformer parameter shapes, and analyze numerical precision requirements.

## Required Work

1. **Stochastic Dynamics & Variance Reduction:**
   - Formalize the variance reduction of the momentum filter:
     $$\text{Var}(M_t) = \frac{1 - \beta}{1 + \beta} \text{Var}(G_t)$$
   - Analyze how the Additive Fiber RMS denominator $D_{ij}(M_t)$ behaves when acting on smoothed velocity $M_t$ versus instantaneous noise $G_t$.
   - Bound expected descent alignment $\mathbb{E}[\langle \nabla \mathcal{L}(\theta), U(M_t) \rangle]$ under standard stochastic Lipschitz assumptions.
2. **Width Scaling & Longest-Fiber Radius:**
   - Formalize the longest-fiber radius normalization:
     $$\rho(m, n) = \sqrt{\max(m, n)}$$
   - Prove that this scaling preserves equal update authority across asymmetric matrices (e.g. embedding tables with large vocabularies $V \times d$, projection heads $d \times V$) without step starvation or gradient explosion.
   - Establish consistent semantics for 1D parameters (RMSNorm gains, biases) through matrixization ($m \times 1$).
3. **Decoupled Weight Decay & Scale Invariance:**
   - Characterize the interaction between decoupled weight decay $W_{t+1} = W_t (1 - \eta \lambda) - \eta U$ and scale-invariant layers (RMSNorm, SwiGLU).
   - Prove that weight decay maintains bounded parameter Frobenius norms $\|W\|_F$, preventing effective step-size decay $\Delta W_{\text{eff}} \to 0$.
4. **Finite Precision Analysis:**
   - Verify safe accumulation precision: establish that row/column RMS and Frobenius norm reductions must accumulate in FP32 to prevent catastrophic cancellation or overflow.
   - Verify that model weights and gradients can safely reside in BF16 or FP32 without loss of stability.

## Gate

Phase 2 passes only if:
- Stochastic descent bounds are mathematically derived under explicit measurable quantities;
- Longest-fiber radius scaling is derived and proven for rectangular and 1D parameters;
- Decoupled weight decay dynamics are formally characterized;
- Numeric precision safety in BF16/FP32 is analytically and empirically confirmed on diagnostics.

Write the standard Phase 2 artifacts (`report.md`, `manifest.json`, `commands.log`, `phases/status/phase2.json`). Commit without force. Do not start Phase 3 in this session.
