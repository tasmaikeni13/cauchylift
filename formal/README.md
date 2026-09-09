# Lean 4 Formal Proof Artifact for CauchyLift

This directory contains formal, machine-checked proofs in Lean 4 verifying the core mathematical properties of the CauchyLift operator.

Pinned toolchain: Lean 4.19.0 with Mathlib 4.19.0 (manifest in `lake-manifest.json`).

```bash
cd formal
lake build
```

## Machine-Checked Properties

* **Degree-0 Scale Invariance (`Basic.lean`):** Proves that the normalized CauchyLift direction is strictly homogeneous of degree zero under scalar multiplication: $U(\alpha M) = U(M)$ for all $\alpha > 0$.
* **Coordinate Magnitude Bounds (`Basic.lean`):** Formally verifies that $|Z_{ij}(M)| \le \min(\sqrt{n}, \sqrt{m})$, guaranteeing that coordinate scaling cannot diverge.
* **Strict Positivity of Fiber RMS Denominator (`CauchyKernel.lean`):** Proves that $D_{ij}(M) = \text{RMS}_{\text{row}, i}(M) + \text{RMS}_{\text{col}, j}(M) > 0$ for all non-zero matrices and active entries.
* **Strict Descent Alignment (`Convergence.lean`):** Machine-checks the inner product positivity $\langle M, U(M) \rangle > 0$.
