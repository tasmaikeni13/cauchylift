# CauchyLift: Cotransverse Rational Gradient Fields for State-Free Matrix Optimization

**Status:** theory-stage manuscript, version 0.1.1, 2026-08-28
**Artifacts:** mathematical Python probes and partial Lean formalization accompany this manuscript
**Empirical status:** no neural-network training has been performed

## Abstract

Matrix optimizers for deep learning commonly obtain their geometry from historical moments, covariance factors, singular vectors, polar factors, norm-ball linear oracles, row/column balancing, or learned rotations. We investigate a different primitive. Given a matrix gradient \(G\), define the energy excluded by entry \((i,j)\) as the sum of all gradient energy outside row \(i\) and all gradient energy outside column \(j\). CauchyLift divides each gradient entry by this **cotransverse energy** and then takes one projective normalization. The map has linear arithmetic cost, requires only reductions and pointwise operations, and stores no persistent optimizer state.

We prove that the resulting direction is scale invariant, odd, sign preserving, permutation and transpose equivariant, and uniformly descent aligned: its cosine with every nonzero exact gradient is at least \(1/\sqrt3\), independent of matrix dimensions. This gives an \(O(T^{-1/2})\) deterministic stationarity bound for smooth nonconvex objectives under normalized steps. On a rank-one gradient, the unnormalized field factorizes through an additive Cauchy kernel; under generic exact conditions this changes rank one to full algebraic rank. A numerical audit shows why this fact must be interpreted cautiously: sampled floating stable rank remains nearly one.

Small quadratic probes show promising direction quality at moderate conditioning and a serious basis/schedule weakness at condition \(10^4\). Consequently, this paper does **not** claim AdamW-like wall time, SOAP/Muon-like convergence, stochastic acceleration, or neural-training efficacy. Its contribution is a new, falsifiable mathematical optimizer hypothesis; a finite literature audit found no close formula through 2026-08-28, but cannot certify universal historical novelty.

## 1. Motivation and research boundary

Adam and AdamW obtain coordinatewise adaptivity from historical first and second moments [1, 2]. Shampoo obtains richer matrix geometry from Kronecker covariance factors and inverse matrix roots [3]. SOAP runs Adam-like adaptation in a changing Shampoo eigenbasis and reports substantial iteration and wall-clock gains on its studied language-model workloads [4]. Muon instead orthogonalizes a momentum matrix through a short Newton–Schulz polynomial, approximating a polar factor [5]. Recent work expands the design space through stateless multi-normalization [7], norm-ball linear minimization oracles [8], row normalization plus whitening [9], structured Fisher approximations [11], operator-norm geometries [12], and adaptive rotations [10].

This progress also makes “new optimizer” an unusually strict research problem. Many visually different update equations reduce to a familiar sequence:

1. accumulate or transform a gradient;
2. rotate, normalize, whiten, or project it;
3. append momentum, Adam, sign, or a scheduler.

The contract for this work forbids that construction. The proposed update may not be Adam plus a matrix statistic, Muon plus a normalization, a rotation plus a base optimizer, a new norm inserted into an existing LMO framework, or a cheap polar approximation. It must be one tensor-valued primitive with a linear, GPU-regular operation graph.

Absolute novelty is not a mathematically certifiable claim: literature search is finite, terminology varies, unpublished work exists, and algebraically equivalent formulas can be written in many forms. We therefore make the narrower reproducible statement:

> In a dated search over primary optimizer papers, their citation neighborhoods, and direct formula queries, we found no optimizer using the normalized field \(G_{ij}/[(S-r_i)+(S-c_j)]\). The search scope and queries are recorded in `research/search_log.md`.

This manuscript distinguishes four evidence levels throughout: proved, machine checked, numerically checked, and hypothesized.

## 2. The cotransverse construction

### 2.1 Notation

Let \(G=(G_{ij})\in\mathbb R^{m\times n}\) be a nonzero exact gradient. Write

\[
S=\lVert G\rVert_F^2,
\qquad
r_i=\sum_{j=1}^nG_{ij}^2,
\qquad
c_j=\sum_{i=1}^mG_{ij}^2.
\]

The usual marginal view focuses on \(r_i\) and \(c_j\). CauchyLift instead assigns entry \((i,j)\) the energy not carried by its row plus the energy not carried by its column:

\[
E_{ij}(G)
= (S-r_i)+(S-c_j)
=2S-r_i-c_j.
\tag{1}
\]

We call \(E(G)\) the **cotransverse energy field**. “Cotransverse” means that both fibers incident to the entry—its row and its column—are excluded. Energy outside both is counted once in each exclusion, because it is transverse to both fibers.

For every \((i,j)\), \(E_{ij}\ge0\). It vanishes at a nonzero entry only when all gradient energy is concentrated at that single entry.

### 2.2 Projective dual

Away from that boundary, define the raw rational field

\[
Z_{ij}(G)=\frac{G_{ij}}{E_{ij}(G)}.
\tag{2}
\]

The numerator is degree one in \(G\), while the denominator is degree two, so \(Z(\alpha G)=\alpha^{-1}Z(G)\). Its magnitude is therefore not an update scale; only its ray is meaningful. Let

\[
\rho_{m,n}=\sqrt{\min(m,n)}.
\]

The CauchyLift direction is

\[
\boxed{
\operatorname{CL}(G)=
\rho_{m,n}\frac{Z(G)}{\lVert Z(G)\rVert_F}
}
\tag{3}
\]

whenever \(Z\) is finite. The radius equals the Frobenius norm of a full-rank rectangular partial isometry. This is a scale convention, not a claim that the direction is orthogonal or polar.

At the one-sparse boundary we define (3) projectively:

\[
\operatorname{CL}(G)
=\lim_{\varepsilon\downarrow0}
\rho_{m,n}
\frac{Z^{(\varepsilon)}(G)}{\lVert Z^{(\varepsilon)}(G)\rVert_F},
\qquad
Z^{(\varepsilon)}_{ij}(G)
=\frac{G_{ij}}{E_{ij}(G)+\varepsilon S}.
\tag{4}
\]

The limit is simply the signed radius on the active entry. The \(\varepsilon\) in (4) defines a boundary limit; it is not an optimizer hyperparameter. For \(G=0\), set \(\operatorname{CL}(0)=0\).

### 2.3 Update rule

For one matrix parameter \(W\), the theory-stage optimizer is only

\[
W_{t+1}=W_t-\eta_t\operatorname{CL}(\nabla f(W_t)).
\tag{5}
\]

There is no momentum, running moment, clipping rule, rotation, whitening pass, matrix root, low-rank projector, trust gate, or fallback optimizer. Weight decay is not part of the primitive.

For a vector, treat it as a \(1\times n\) matrix; then \(E_{1j}=S-G_{1j}^2\). A scalar uses the boundary rule. Higher-order tensors can be flattened along their semantic output axis versus all remaining axes, but this choice requires an architecture study and is not analyzed here.

### 2.4 Pseudocode

```text
input: nonzero G in R^(m x n), scalar step eta

Q <- G^2                                      # pointwise, FP32 accumulation
S <- sum(Q)
r <- row_sum(Q)
c <- col_sum(Q)
E_ij <- (S - r_i) + (S - c_j)                # broadcast

if an active E_ij is exactly zero:
    Z <- projective boundary direction
else:
    Z_ij <- G_ij / E_ij

D <- sqrt(min(m,n)) * Z / FrobeniusNorm(Z)
W <- W - eta * D
```

The reference Python implementation rescales \(G\) before squaring and multiplies all reciprocals by the smallest positive denominator. Both changes cancel under projective normalization and avoid overflow without changing the mathematical direction.

## 3. Why this is not row/column normalization

Direct marginal methods usually divide by, normalize toward, or accumulate a function of \(r_i\) and \(c_j\). Sinkhorn-style methods repeatedly rescale rows and columns until a balance condition is approached [7]. Factorized adaptive methods approximate an entrywise statistic by a separable row factor times a column factor. CauchyLift instead uses

\[
\frac{1}{(S-r_i)+(S-c_j)}.
\]

Three algebraic differences matter.

1. **Complement rather than marginal.** Increasing \(r_i+c_j\) decreases the denominator, so the field emphasizes an energetic row-column intersection. Direct normalization normally suppresses that intersection.
2. **Nonseparability.** In general,
   \(1/(2S-r_i-c_j)\ne a_i b_j\). The map is not one left diagonal scaling followed by one right diagonal scaling.
3. **No balancing fixed point.** Equation (3) is evaluated once. It does not seek prescribed row or column norms.

It is still possible to write \(\operatorname{CL}(G)=P(G)G\) with a diagonal, gradient-dependent \(P\). That representation is not a meaningful collision test: every sign-preserving direction admits it. The relevant novelty object is the coupled cotransverse field itself.

## 4. Elementary structure

### Proposition 1 — symmetries and fixed radius

For every nonzero matrix \(G\), every \(\alpha\ne0\), row permutation \(P\), and column permutation \(Q\),

\[
\begin{aligned}
\operatorname{CL}(\alpha G)&=\operatorname{sign}(\alpha)\operatorname{CL}(G),\\
\operatorname{CL}(-G)&=-\operatorname{CL}(G),\\
\operatorname{CL}(PGQ)&=P\operatorname{CL}(G)Q,\\
\operatorname{CL}(G^\top)&=\operatorname{CL}(G)^\top,\\
\lVert\operatorname{CL}(G)\rVert_F&=\rho_{m,n}.
\end{aligned}
\tag{6}
\]

Moreover, every nonzero output entry has the same sign as its input entry.

**Proof.** Squared energy scales by \(\alpha^2\), so the raw field scales by \(1/\alpha\); normalization removes \(|\alpha|^{-1}\) and preserves its sign. Row/column permutations merely permute the corresponding marginal sums, and transpose exchanges them. The remaining statements follow directly from (2)–(4). ∎

The map is not orthogonally equivariant. That is intentional: recent symmetry-compatible optimizer work places bi-orthogonally equivariant general-matrix updates in the spectral family, the neighborhood this project seeks to leave [22].

## 5. A dimension-independent descent angle

The reciprocal in (2) looks dangerous: a small denominator might redirect the update away from the gradient. The following theorem gives a global safety result without clipping or blending.

### Theorem 1 — uniform alignment

For every nonzero \(G\in\mathbb R^{m\times n}\),

\[
\boxed{
\frac{\langle G,\operatorname{CL}(G)\rangle}
{\lVert G\rVert_F\lVert\operatorname{CL}(G)\rVert_F}
\ge \frac1{\sqrt3}
}
\tag{7}
\]

and therefore \(\langle G,\operatorname{CL}(G)\rangle>0\).

#### Proof

The one-sparse boundary has cosine one, so consider positive denominators. Normalize energy shares:

\[
a_{ij}=\frac{G_{ij}^2}{S},\qquad
R_i=\sum_j a_{ij},\qquad
C_j=\sum_i a_{ij},\qquad
h_{ij}=\frac{E_{ij}}S=2-R_i-C_j.
\]

Then \(a_{ij}\ge0\) and \(\sum_{ij}a_{ij}=1\). The energy in the union of row \(i\) and column \(j\) is \(R_i+C_j-a_{ij}\le1\), hence

\[
h_{ij}=2-R_i-C_j\ge1-a_{ij}.
\tag{8}
\]

Also \(h_{ij}\le2\). Put \(w_{ij}=h_{ij}^{-1}\), and define

\[
A=\sum_{ij}a_{ij}w_{ij},
\qquad
B=\sum_{ij}a_{ij}w_{ij}^2.
\]

The squared cosine between \(G\) and \(Z\), and therefore between \(G\) and \(\operatorname{CL}(G)\), is \(A^2/B\). Since \(h_{ij}\le2\), we have \(A\ge1/2\). From (8), \(h_{ij}+a_{ij}\ge1\), so

\[
w_{ij}\le1+a_{ij}w_{ij}.
\]

Consequently,

\[
\begin{aligned}
B
&=\sum_{ij}a_{ij}w_{ij}^2\\
&\le \sum_{ij}a_{ij}w_{ij}
   +\sum_{ij}a_{ij}^2w_{ij}^2\\
&\le A+A^2\\
&\le3A^2,
\end{aligned}
\]

where the penultimate inequality uses
\(\sum x_k^2\le(\sum x_k)^2\) for nonnegative \(x_k=a_{ij}w_{ij}\), and the last uses \(A\ge1/2\). Thus \(A^2/B\ge1/3\). ∎

The bound is conservative. Across 5,000 seeded random matrices spanning shapes from \(1\times1\) to \(17\times5\), zeros, signs, and roughly 20 orders of magnitude of scaling, the smallest observed cosine was 0.9242. A numerical observation is not a sharper theorem.

## 6. Deterministic convergence

### Theorem 2 — normalized smooth descent

Let \(f:\mathbb R^{m\times n}\to\mathbb R\) be differentiable, \(L\)-smooth in Frobenius norm, and bounded below by \(f_\inf\). Let \(G_t=\nabla f(W_t)\), \(D_t=\operatorname{CL}(G_t)\), and use (5) with a constant \(\eta>0\). If \(G_t\ne0\) for \(t<T\), then

\[
\min_{0\le t<T}\lVert G_t\rVert_F
\le
\frac{\sqrt3\,[f(W_0)-f_\inf]}{\eta\rho_{m,n}T}
+\frac{\sqrt3}{2}L\eta\rho_{m,n}.
\tag{9}
\]

Choosing

\[
\eta=\sqrt{\frac{2[f(W_0)-f_\inf]}{L\rho_{m,n}^2T}}
\]

gives

\[
\boxed{
\min_{0\le t<T}\lVert\nabla f(W_t)\rVert_F
\le
\sqrt{\frac{6L[f(W_0)-f_\inf]}T}.
}
\tag{10}
\]

#### Proof

Smoothness, Theorem 1, and \(\lVert D_t\rVert_F=\rho_{m,n}\) give

\[
f(W_{t+1})
\le f(W_t)
-\frac{\eta\rho_{m,n}}{\sqrt3}\lVert G_t\rVert_F
+\frac{L\eta^2\rho_{m,n}^2}{2}.
\]

Sum from \(t=0\) to \(T-1\), lower-bound the final objective by \(f_\inf\), divide by \(T\), and minimize the resulting two-term bound over \(\eta\). ∎

This is a safety theorem, not an acceleration theorem. It uses exact gradients and the standard normalized-gradient rate. Because \(G\mapsto\operatorname{CL}(G)\) is nonlinear, an unbiased stochastic gradient does not imply an unbiased transformed direction. An unconditional stochastic theorem remains open.

## 7. The Cauchy law on rank-one gradients

The optimizer's name comes from an exact structural identity rather than an imported Cauchy-noise model.

### Theorem 3 — rank-one Cauchy factorization

Let \(G=uv^\top\), with nonzero \(u\in\mathbb R^m\) and \(v\in\mathbb R^n\). Put

\[
U=\lVert u\rVert_2^2,
\quad V=\lVert v\rVert_2^2,
\quad a_i=\frac{u_i^2}{U},
\quad b_j=\frac{v_j^2}{V},
\quad x_i=1-a_i,
\quad y_j=1-b_j.
\]

Away from the one-sparse boundary,

\[
Z(G)=\frac1{UV}\operatorname{diag}(u),C(x,y)\,\operatorname{diag}(v),
\qquad
C(x,y)_{ij}=\frac1{x_i+y_j}.
\tag{11}
\]

**Proof.** The total, row, and column energies are \(UV\), \(u_i^2V\), and \(v_j^2U\). Therefore

\[
E_{ij}=UV[(1-a_i)+(1-b_j)]=UV(x_i+y_j),
\]

and substitution into (2) gives (11). ∎

### Corollary 3.1 — generic algebraic rank lift

If every entry of \(u\) and \(v\) is nonzero, the values \(u_i^2\) are pairwise distinct, the values \(v_j^2\) are pairwise distinct, and the denominators in (11) are positive, then

\[
\operatorname{rank}\operatorname{CL}(uv^\top)=\min(m,n).
\tag{12}
\]

**Proof sketch.** The all-entries-nonzero hypotheses make both diagonal factors injective on every selected square submatrix, and scalar normalization preserves rank. Every square submatrix of \(C\) with distinct \(x_i\) and \(y_j\) is a Cauchy matrix. Its determinant is a nonzero product of pairwise differences divided by \(\prod_{ij}(x_i+y_j)\). ∎

The requirement that every factor entry be nonzero is essential. For example, \(u=(1,0)^\top\) and \(v=(1,2)^\top\) have distinct nonzero squared magnitudes, but the zero row remains zero after the transform and the output rank is one. The Phase 1 adversarial suite retains this counterexample because the earlier wording omitted the all-entries-nonzero hypothesis.

For \(2\times2\), the identity is explicit:

\[
\det C
=\frac{(x_1-x_2)(y_1-y_2)}
{(x_1+y_1)(x_1+y_2)(x_2+y_1)(x_2+y_2)}.
\tag{13}
\]

The Lean artifact checks the two-by-two rational identity and nondegeneracy assumptions.

### 7.1 Why algebraic rank is not the proposed acceleration mechanism

Exact rank is discontinuous and can be created by arbitrarily small singular values. In the included exact rational example, a rank-one \(4\times4\) outer product maps to exact rank four. Yet for 200 random log-normal factor pairs at each of sizes 4, 8, 16, and 32, median output stable rank was only about 1.00005–1.00017. The algebraic result therefore does **not** establish useful spectral diversity.

The remaining performance hypothesis is more modest: reciprocal cotransverse energy creates a dense smooth analogue of greedy mode emphasis. When one row and one column jointly carry much of the gradient energy, their intersection receives the largest multiplier. If an accurate scalar step removes that dominant curvature mode, later directions can expose weaker modes. The hard-condition scheduled probes show exactly why this hypothesis may fail without reliable step control.

## 8. Complexity and implementation model

For an \(m\times n\) gradient, CauchyLift needs:

- one elementwise square;
- a total sum and row/column sums;
- one broadcasted denominator and reciprocal multiplication;
- one Frobenius-norm reduction;
- one scalar multiply and parameter update.

Arithmetic work is \(O(mn)\). Parallel reduction depth is logarithmic in the reduced dimensions. Persistent optimizer state is zero; transient row/column arrays require \(O(m+n)\) storage, or can be managed inside fused kernels. There are no matrix-matrix products, decompositions, iterative inverse roots, QR factorizations, or historical tensors.

| Method family | Persistent state beyond parameters/gradient | Dominant optimizer operations | Abstract arithmetic class |
|---|---:|---|---:|
| AdamW | First and second moment tensors | Pointwise passes | \(O(mn)\) |
| Shampoo/SOAP | Matrix statistics plus element state, implementation dependent | Covariances, eigensolvers/inverse roots, transforms | More than pointwise linear work |
| Muon | Usually momentum tensor | Several matrix multiplications for Newton–Schulz | Shape-dependent GEMMs |
| SinkGD | None | Repeated row/column normalization | \(O(Lmn)\) for \(L\) rounds |
| CauchyLift | None | Two marginal reductions + pointwise rational field | \(O(mn)\) |

This operation graph is GPU-friendly in the limited algorithmic sense of regular dense reads, reductions, and pointwise arithmetic. No kernel has been implemented, so no wall-clock claim is made. Division, extra reductions, launch overhead, and memory traffic could erase the abstract advantage.

## 9. Numerical methodology

All scripts use the Python standard library. They are mathematical probes, not model training.

### 9.1 Property checks

`run_property_checks.py` generated 5,000 matrices using a fixed seed, seven shapes, random zeros and signs, and log-uniform magnitudes. It checked:

- fixed output norm;
- positive-scale invariance and oddness;
- transpose equivariance;
- strict positive inner product;
- the \(1/\sqrt3\) cosine floor.

All checks passed. Maximum reported equivariance and norm errors were below \(9\times10^{-16}\). This checks the implementation, not the proof.

### 9.2 Quadratic direction suite

For \(W\in\mathbb R^{4\times4}\), the suite uses

\[
f(W)=\frac12\langle W,H_LWH_R\rangle_F,
\qquad
\nabla f(W)=H_LWH_R,
\tag{14}
\]

with diagonal or randomly rotated positive-definite factors. Two Kronecker-Hessian condition numbers are used: \(10^2\) and \(10^4\). Sixteen problem instances are split equally into schedule tuning and held-out reporting.

Two tests answer different questions:

1. **Exact line search:** isolates direction quality; target relative objective \(10^{-8}\), cap 600 steps.
2. **Scheduled step:** \(\eta/\sqrt t\) for 400 steps; \(\eta\) is selected on the tuning half and reported on the held-out half.

Baselines are normalized gradient, sign, five-round Sinkhorn normalization, and an exact-polar direction computed by a small Jacobi eigensolver. The exact polar is a mathematical reference, not a runtime Muon implementation.

#### Median results

Entries are `exact-line iterations / held-out log10 relative objective after 400 scheduled steps`. A value of 600 can be a capped failure.

| Hessian condition | Geometry | Normalized gradient | Sign | Sinkhorn-5 | Exact polar | CauchyLift |
|---:|---|---:|---:|---:|---:|---:|
| \(10^2\) | axis aligned | 267.5 / −5.285 | 109 / −4.915 | 37 / −5.722 | 327 / −5.112 | **39.5 / −6.612** |
| \(10^2\) | rotated | 306.5 / −5.246 | 571.5 / −4.037 | 201.5 / −4.761 | 323 / −3.088 | **78 / −5.263** |
| \(10^4\) | axis aligned | 600 / −3.150 | 600 / **−5.265** | 317 / **−5.922** | 600 / −4.018 | **182 / −2.964** |
| \(10^4\) | rotated | 600 / −3.356 | 600 / −2.998 | 600 / −3.307 | 600 / **−4.640** | 600 / −3.078 |

At condition \(10^2\), CauchyLift is strong under both probes. At condition \(10^4\), it retains good axis-aligned exact-line direction quality but performs poorly with the scheduled step, and every rotated exact-line run hits the cap. This is evidence of step sensitivity and basis dependence, not evidence of SOAP/Muon-like convergence.

### 9.3 Rejected-candidate checks

The exterior/cofactor candidate's condition map matches a Halley triple-angle identity to numerical precision. The cross-ratio plaquette dual fails 198 of 200 exact-line probes at the 1,000-step cap, versus 46 for sign. These results are kept so that rejected ideas are not later rediscovered and marketed under new names.

## 10. Closest work and novelty analysis

### Historical moments

Adam-family methods transform each coordinate using historical moments [1, 2]. CauchyLift is instantaneous and nonseparably couples an entry to two complement sums. Adding Adam moments to it would be a different, forbidden hybrid.

### Matrix preconditioners and polar methods

Shampoo, SOAP, Muon, PRISM, Dion, and related methods use covariances, basis changes, matrix polynomials, polar factors, or low-rank subspaces [3–5, 15, 16]. CauchyLift computes none of these. The discarded cofactor branch did reduce to polar/Halley behavior, which is why it was rejected.

### Norm geometry and multi-normalization

Scion derives optimizer directions from norm-constrained linear minimization [8]. SinkGD and related work alternate direct row and column normalization [7]. CauchyLift is not presented as a new norm in an existing oracle and does not balance marginals. Its complement denominator is, however, closest in implementation pattern to stateless row/column methods. This is the highest residual collision risk.

### Structured Fisher and marginal scaling

RACS/Alice and operator-norm analyses show that row/column information already supports powerful structured rules [11, 12]. CauchyLift's distinction is not “using rows and columns”; that is established. It is the one-pass reciprocal of the **sum of their excluded energies**, which is nonseparable and produces the Cauchy law (11).

### Rotations

ARO treats rotation as a first-class state and demonstrates that many matrix optimizers can be described as rotated base maps [10]. CauchyLift has no rotation. Its poor hard rotated-quadratic result means a rotation wrapper is tempting, but adding one would violate the research contract rather than validate the primitive.

## 11. Limitations

1. **No training evidence.** The central performance target is untested.
2. **No wall-clock evidence.** \(O(mn)\) does not imply AdamW wall time.
3. **No unconditional stochastic theory.** Nonlinear transformation introduces bias.
4. **Basis dependence.** Only permutations and transpose are symmetries; arbitrary rotations are not.
5. **Step sensitivity.** The condition-\(10^4\) scheduled result is a direct warning.
6. **Algebraic versus numerical rank.** Generic full exact rank can carry negligible extra singular mass.
7. **Boundary behavior.** The projective one-sparse limit is exact, but near-boundary finite-precision behavior needs kernel-level study.
8. **Radius scaling.** \(\sqrt{\min(m,n)}\) is a principled convention, not a neural-architecture theorem.
9. **Finite novelty search.** No search can prove that an equivalent unpublished or differently named method does not exist.
10. **No acceleration theorem.** The stationarity result matches normalized first-order order, with a constant-factor alignment loss.

## 12. Falsifiable next phase

The next phase should implement only the primitive in (1)–(5), with no momentum or rescue mixture. At equal tuning budgets it should be compared with AdamW, Muon, SOAP, SinkGD, normalized gradient, and sign across predeclared language, vision, and non-square-tensor workloads. Primary metrics are tokens/examples to target, validation quality, optimizer-only time, full-step time, memory, update concentration, stable rank, and loss spikes.

The hypothesis should be rejected if it requires a conventional optimizer component to become stable or competitive, exceeds AdamW optimizer-step time by more than 15% after fusion, or fails the predeclared tokens-to-target criterion on most workloads. The full protocol is in `research/future_experiment_protocol.md`.

## 13. Conclusion

CauchyLift proposes a mathematical direction absent from the searched optimizer families: projective dualization by cotransverse row-column energy. The primitive is simple enough for linear reduction-only execution yet structured enough to yield a nontrivial uniform-angle theorem and an exact Cauchy-kernel law. Those are genuine theoretical results. They do not establish the desired empirical outcome. The hard quadratic and stable-rank negatives narrow the mechanism and make the next experiment decisive.

The appropriate current claim is therefore precise: **CauchyLift is a new, scoped-search-distinct, formally analyzable optimizer hypothesis—not a demonstrated replacement for AdamW, SOAP, or Muon.**

## References

1. D. P. Kingma and J. Ba. [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980). 2014.
2. I. Loshchilov and F. Hutter. [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101). 2017.
3. V. Gupta, T. Koren, and Y. Singer. [Shampoo: Preconditioned Stochastic Tensor Optimization](https://proceedings.mlr.press/v80/gupta18a.html). ICML 2018.
4. N. Vyas et al. [SOAP: Improving and Stabilizing Shampoo using Adam](https://arxiv.org/abs/2409.11321). 2024/2025.
5. G. Y. Kim and M. Oh. [Convergence of Muon with Newton–Schulz](https://arxiv.org/abs/2601.19156). ICLR 2026.
6. T. Do, S. Dereich, and A. Jentzen. [On MUON Optimization: From Non-convergence to an Error Analysis with Polar Express and the Newton–Schulz Polynomial from Implementations](https://arxiv.org/abs/2608.04607). 2026.
7. M. Scetbon, C. Ma, W. Gong, and E. Meeds. [Gradient Multi-Normalization for Stateless and Scalable LLM Training](https://arxiv.org/abs/2502.06742). 2025.
8. T. Pethick et al. [Training Deep Learning Models with Norm-Constrained LMOs](https://arxiv.org/abs/2502.07529). 2025.
9. C. Ma et al. [SWAN: Preprocessing SGD Enables Adam-Level Performance On LLM Training With Significant Memory Reduction](https://proceedings.mlr.press/v267/ma25g.html). ICML 2025.
10. [ARO: A New Lens On Matrix Optimization For Large Models](https://arxiv.org/abs/2602.09006). 2026.
11. [Towards Efficient Optimizer Design for LLM via Structured Fisher Approximation with a Low-Rank Extension](https://arxiv.org/abs/2502.07752). 2025.
12. [On the Width Scaling of Neural Optimizers Under Matrix Operator Norms I](https://arxiv.org/abs/2603.09952). 2026.
13. [What Really Matters in Matrix-Whitening Optimizers?](https://arxiv.org/abs/2510.25000). 2025.
14. [NorMuon: Making Muon More Efficient and Scalable](https://arxiv.org/abs/2510.05491). 2025.
15. [Dion: Distributed Orthonormalized Updates](https://arxiv.org/abs/2504.05295). 2025.
16. Y. Yang. [PRISM: Structured Optimization via Anisotropic Spectral Shaping](https://arxiv.org/abs/2602.03096). 2026.
17. [OLion](https://arxiv.org/abs/2602.01105). 2026.
18. [MuonBP](https://arxiv.org/abs/2510.16981). 2025.
19. [Fantastic Pretraining Optimizers and Where to Find Them](https://arxiv.org/abs/2509.02046). 2025.
20. [A Large Batch Optimizer Reality Check](https://arxiv.org/abs/2102.06356). 2021.
21. [XGrad: Boosting Gradient-Based Optimizers With Weight Prediction](https://arxiv.org/abs/2305.18240). 2023.
22. T. T.-K. Lau and W. Su. [Symmetry-Compatible Principle for Optimizer Design: Embeddings, LM Heads, SwiGLU MLPs, and MoE Routers](https://arxiv.org/abs/2605.18106). 2026.
