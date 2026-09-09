# CauchyLift Research Contract

**Status:** Active Research Specification (v1.0.0)  
**Author:** Tasmai Keni  
**Hardware Target:** AMD Instinct MI300X (`gfx942`, ROCm 10.0)

---

## 1. Objective

Develop and validate **CauchyLift**, a curvature-adaptive matrix optimizer for deep neural networks that achieves:
1. **Curvature Adaptation:** Automatic layer-wise and coordinate-wise Riemannian adaptation via an Additive Fiber RMS Cauchy Operator ($D_{ij} = \text{RMS}_{\text{row}, i} + \text{RMS}_{\text{col}, j}$).
2. **Minimal State Footprint:** Exactly one momentum state tensor per parameter (4 bytes/param in FP32, 2 bytes/param in BF16), cutting optimizer state memory by **50% compared to AdamW**.
3. **Linear Arithmetic Complexity:** Strictly $O(N^2)$ reductions and element-wise arithmetic, completely avoiding $O(N^3)$ matrix multiplications, polar decompositions (Newton–Schulz iterations), or matrix inversions.
4. **Sub-Millisecond Kernel Latency:** Native fused ROCm/HIP multi-tensor kernel dispatch executing in $<0.8$ ms on AMD Instinct MI300X ($>15\times$ faster than Muon).
5. **Decoupled Parameter Regulation:** Direct weight shrinkage $W_{t+1} = W_t(1 - \eta \lambda) - \eta U_t$ preserving representation capacity in scale-invariant Pre-RMSNorm Transformers.

---

## 2. Defining Equations

$$M_t = \beta M_{t-1} + (1 - \beta) G_t \quad (\beta = 0.95)$$
$$D_{ij}(M_t) = \text{RMS}(M_{t, i, :}) + \text{RMS}(M_{t, :, j}) = \sqrt{\frac{1}{n} \sum_{k=1}^n M_{ik}^2} + \sqrt{\frac{1}{m} \sum_{l=1}^m M_{lj}^2}$$
$$Z_{ij}(M_t) = \frac{M_{t, ij}}{D_{ij}(M_t)}, \qquad U(M_t) = \sqrt{\max(m, n)} \frac{Z(M_t)}{\|Z(M_t)\|_F}$$
$$W_{t+1} = W_t (1 - \eta_t \lambda) - \eta_t U(M_t)$$

---

## 3. Verified Mathematical Guarantees

* **Theorem 1 (Degree-0 Scale Invariance):** $U(\alpha M) = U(M)$ for all $\alpha > 0$.
* **Theorem 2 (Coordinate Magnitude Bounds):** $|Z_{ij}(M)| \le \min(\sqrt{n}, \sqrt{m})$ for all active coordinates.
* **Theorem 3 (Strict Descent Alignment):** $\langle M, U(M) \rangle_F > 0$ for all non-zero momentum matrices $M$.
* **Proposition 1 (Stochastic Variance Reduction):** $\text{Var}(M_t) = \frac{1-\beta}{1+\beta}\text{Var}(G_t) \approx \frac{1}{39}\text{Var}(G_t)$ for $\beta=0.95$.
