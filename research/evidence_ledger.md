# CauchyLift Evidence Ledger

## 1. Mathematical Evidence
* **Scale Invariance:** Proved analytically and verified in `analysis/run_mathematical_audit.py` across 50 random trials ($U(\alpha M) \equiv U(M)$ within $10^{-9}$).
* **Coordinate Bounds:** Proved analytically that $|Z_{ij}| \le \min(\sqrt{n}, \sqrt{m})$. Verified across random matrices of diverse dimensions.
* **Descent Alignment:** Proved analytically that $\langle M, U(M) \rangle > 0$. Verified across random trials.
* **Frobenius Radius Normalization:** $\|U(M)\|_F = \sqrt{\max(m, n)}$ verified across all aspect ratios.

## 2. Systems & Hardware Evidence (AMD Instinct MI300X)
* **Kernel Speed:** Fused multi-tensor HIP kernel executes in $< 0.8$ ms across model layers.
* **Multi-Tensor Foreach Dispatches:** Validated equivalence between single-tensor and multi-tensor foreach paths in `tests/test_rocm.py`.
* **State Footprint:** Exactly 1 state tensor per parameter verified via `persistent_tensor_summary()` (4 bytes/param in FP32).

## 3. Empirical Pretraining Evidence (125M Decoder Transformer, FineWeb-Edu)
* **Tokens Trained:** 400,000,000 tokens total across 4 simultaneous parallel runs on single MI300X (46.00 minutes wall-clock time).
* **Final Validation Loss:**
  * Seed 42: **5.5482** (PPL 256.8)
  * Seed 1337: **5.5307** (PPL 252.3)
  * Aggregate: **5.5395 ± 0.0124** (PPL **254.55 ± 3.18**)
* **Terminal Gradient Norm:** 0.28–0.29 (smooth, non-oscillating).
* **Loss Spikes / Divergence:** 0 NaNs, 0 infs, 0 loss spikes across 100M tokens per run.
