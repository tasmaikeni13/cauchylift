# CauchyLift: Cotransverse Rational Gradient Fields for State-Free Matrix Optimization

**Status:** theory-stage manuscript, version 0.2.0, 2026-08-29
**Artifacts:** mathematical Python probes and partial Lean formalization accompany this manuscript
**Empirical status:** no neural-network training has been performed; the single-MI300X Phase 3 correctness and optimizer-step speed gates pass, while independent second-family replication remains open

## Abstract

Matrix optimizers for deep learning commonly obtain their geometry from historical moments, covariance factors, singular vectors, polar factors, norm-ball linear oracles, row/column balancing, or learned rotations. We investigate a different primitive. Given a matrix gradient \(G\), define the energy excluded by entry \((i,j)\) as the sum of all gradient energy outside row \(i\) and all gradient energy outside column \(j\). CauchyLift divides each gradient entry by this **cotransverse energy** and then takes one projective normalization. The map has linear arithmetic cost, requires only reductions and pointwise operations, and stores no persistent optimizer state.

We prove that the resulting direction is scale invariant, odd, sign preserving, permutation and transpose equivariant, and uniformly descent aligned: its cosine with every nonzero exact gradient is strictly greater than \(1/\sqrt3\), independent of matrix dimensions. This gives an \(O(T^{-1/2})\) deterministic stationarity bound for smooth nonconvex objectives under normalized steps. We also prove continuity at every one-sparse boundary with an explicit \(O(\tau^{3/2})\) modulus in off-cell energy \(\tau\), an interior Lipschitz bound, and conditional expected stochastic descent when minibatch noise is small relative to the true gradient. A scalar unbiased-noise counterexample shows why the condition cannot simply be discarded.

On a rank-one gradient, the unnormalized field factorizes through an additive Cauchy kernel; under generic exact conditions this changes rank one to full algebraic rank. A numerical audit shows why this fact must be interpreted cautiously: sampled floating stable rank remains nearly one. On an isolated two-mode diagonal quadratic, exact line search maps the gradient ratio \(q\) to \(-q^{-3}\), versus \(-q^{-1}\) for normalized gradient and unit magnitude for sign, row/column-normalized, and polar controls. This is a sharply testable mode-alternation signature, not an acceleration theorem.

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

whenever \(Z\) is finite. The radius is the maximal transpose-symmetric choice for which both the average squared row norm and average squared column norm of every update are at most one. Indeed, a radius \(\rho\) gives averages \(\rho^2/m\) and \(\rho^2/n\), so both fiber constraints require \(\rho^2\le\min(m,n)\). This also equals the Frobenius norm of a full-rank rectangular partial isometry, but no orthogonal or polar structure is claimed.

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

The same primitive is total on every trainable decoder parameter shape. A scalar is reshaped to \(1\times1\); a vector of length \(d\) to \(d\times1\); a stored matrix keeps its row-column shape; and a tensor of rank at least three is reshaped to \(d_0\times\prod_{k>0}d_k\), with the first axis serving as its semantic output or record axis. Transposition does not change the direction after mapping back or the radius, so the vector orientation is immaterial.

For the initial bias-free decoder contract, token embeddings and output heads use their stored vocabulary-by-hidden shape; tied weights are transformed once after their shared gradient is accumulated. Attention and MLP weights use output-by-input storage. RMS-normalization gains use \(d\times1\). Scalars use the boundary rule. If a later architecture has biases, each bias uses the same vector rule; it does not receive Adam or SGD. Sparse layouts must implement the mathematically identical support-aware field or materialize the dense gradient—never switch optimizer families.

### 2.4 Pseudocode

```text
input: finite G in R^(m x n), scalar step eta

if all(G == 0): return 0
if exactly_one_nonzero(G): return signed_boundary_direction

Gq <- G / max(abs(G))                         # projectively neutral
Q <- Gq^2                                     # pointwise, FP32 accumulation
r <- row_sum(Q)
c <- col_sum(Q)
outside_row_i <- exclusion_sum(r, i)          # nonnegative prefix/suffix sums
outside_col_j <- exclusion_sum(c, j)
E_ij <- outside_row_i + outside_col_j         # no dominant-total subtraction
e_min <- min(E_ij over active entries)
Z_ij <- Gq_ij * e_min / E_ij                  # bounded ray representative

D <- sqrt(min(m,n)) * Z / FrobeniusNorm(Z)
W <- W - eta * D
```

The reference Python implementation follows this exclusion-safe form. Rescaling \(G\) before squaring and multiplying all reciprocals by the smallest positive active denominator both cancel under projective normalization. They prevent overflow without adding a stabilizer or changing the mathematical direction.

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
> \frac1{\sqrt3}
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

### 5.1 Equality and boundary strata

The bound in (7) has no equality case. Away from the one-sparse boundary, every active cell has \(R_i+C_j>0\), so \(h_{ij}<2\) and therefore its active reciprocal weight is strictly greater than \(1/2\). Hence \(A>1/2\), making the closing inequality strict if equality at \(1/\sqrt3\) were attempted. At the one-sparse boundary the cosine is one. The best global constant may be larger than \(1/\sqrt3\); this work does not claim a sharp replacement.

The normalized-energy simplex also resolves boundary continuity. Let \(X=G/\|G\|_F\), let cell \(p\) have energy share \(a_p=1-\tau\), and let \(e_p\) denote its signed coordinate direction. For \(0<\tau<1\), the dominant denominator obeys

\[
\tau\le h_p\le2\tau,
\]

because every off-cell energy term lies outside at least one of the dominant cell's two fibers and outside at most both. For every other cell \(q\), the union bound gives \(h_q\ge1-a_q\ge1-\tau\). If \(F_{ij}=X_{ij}/h_{ij}\), then

\[
\frac{\|F_{-p}\|_F}{|F_p|}
\le
2\left(\frac{\tau}{1-\tau}\right)^{3/2}.
\]

Normalizing a vector whose transverse-to-dominant norm ratio is \(u\) changes it from the dominant coordinate by at most \(u\). Consequently,

\[
\boxed{
\left\|\frac{\operatorname{CL}(G)}{\rho_{m,n}}-e_p\right\|_F
\le2\left(\frac{\tau}{1-\tau}\right)^{3/2}.
}
\tag{8a}
\]

Thus the projective extension is continuous on every one-sparse stratum and is cubic in off-cell amplitude \(\sqrt\tau\). The active denominator ratio is at most \(2/\tau\), and the examples in `boundary_suite.json` show \(\Theta(1/\tau)\) growth, so the raw field is ill-conditioned even while its normalized ray is stable.

### 5.2 Interior sensitivity

On the unit sphere, restrict to a region where every normalized denominator satisfies \(h_{ij}\ge\delta>0\). Row and column energy differences obey

\[
\|h(X)-h(Y)\|_\infty\le4\|X-Y\|_F.
\]

For \(F(X)=X/h(X)\), entrywise division and \(\|Y\|_F=1\) give

\[
\|F(X)-F(Y)\|_F
\le(\delta^{-1}+4\delta^{-2})\|X-Y\|_F.
\]

Because \(h\le2\), \(\|F(X)\|_F\ge1/2\). The standard normalization inequality therefore yields

\[
\boxed{
\left\|\frac{\operatorname{CL}(X)}\rho-
\frac{\operatorname{CL}(Y)}\rho\right\|_F
\le(4\delta^{-1}+16\delta^{-2})\|X-Y\|_F.
}
\tag{8b}
\]

This is an explicit regional Lipschitz bound, not a claim that its constant is sharp. There is no continuous extension at the zero matrix: two different rays \(tX\) and \(tY\) both approach zero while their scale-invariant outputs remain separated. The zero-gradient value in (4) is therefore an algorithmically total convention, not topological continuity at zero.

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

This is a safety theorem, not an acceleration theorem. It uses exact gradients and the standard normalized-gradient rate.

### 6.1 Conditional stochastic alignment

Let \(\mu=\nabla f(W)\), let \(g\) be an integrable stochastic gradient with \(\mathbb E g=\mu\), and put \(D=\operatorname{CL}(g)\). Write \(\gamma=1/\sqrt3\) and \(\rho=\rho_{m,n}\). The deterministic angle theorem and Cauchy--Schwarz give the pointwise inequality

\[
\langle\mu,D\rangle
=\langle g,D\rangle+\langle\mu-g,D\rangle
\ge\rho\left(\gamma\|g\|_F-\|g-\mu\|_F\right),
\tag{10a}
\]

where the inequality remains valid when \(g=0\) because then \(D=0\). Therefore

\[
\mathbb E\langle\mu,D\rangle
\ge\rho\left(\gamma\mathbb E\|g\|_F-
\mathbb E\|g-\mu\|_F\right).
\tag{10b}
\]

If \(\sigma^2=\mathbb E\|g-\mu\|_F^2<\infty\), Jensen and Cauchy--Schwarz yield the more directly interpretable bound

\[
\boxed{
\mathbb E\langle\mu,D\rangle
\ge\rho(\gamma\|\mu\|_F-\sigma).
}
\tag{10c}
\]

Thus \(\sigma<\gamma\|\mu\|_F\) is a sufficient signal-to-noise condition for positive expected alignment. For an \(L\)-smooth objective,

\[
\mathbb E f(W-\eta D)
\le f(W)-\eta\rho(\gamma\|\mu\|_F-\sigma)
+\frac{L\eta^2\rho^2}{2},
\tag{10d}
\]

so strict one-step expected descent follows when the margin is positive and

\[
0<\eta<\frac{2(\gamma\|\mu\|_F-\sigma)}{L\rho}.
\]

A high-probability alternative is also explicit. If \(\|g-\mu\|_F\le\kappa\|\mu\|_F\) with probability at least \(1-\zeta\), then bounding the remaining event by \(-\rho\|\mu\|_F\) gives

\[
\mathbb E\langle\mu,D\rangle
\ge\rho\|\mu\|_F
\left[(1-\zeta)(\gamma-(1+\gamma)\kappa)-\zeta\right].
\tag{10e}
\]

These conditions use quantities estimable during training: a large-batch proxy for \(\|\mu\|_F\), repeated microbatch deviations for \(\sigma\), or an empirical relative-error quantile for \((\kappa,\zeta)\). They are sufficient, not claimed necessary.

On an interior region with normalized denominators at least \(\delta\), (8b) also controls transformation bias. If \(g\ne0\) almost surely and \(\mu\ne0\),

\[
\|\mathbb E D-\operatorname{CL}(\mu)\|_F
\le
2\rho(4\delta^{-1}+16\delta^{-2})
\frac{\mathbb E\|g-\mu\|_F}{\|\mu\|_F}.
\tag{10f}
\]

Unbiasedness alone is decisively insufficient. In one dimension, let \(g=10\) with probability \(0.1\) and \(g=-0.5\) with probability \(0.9\). Then \(\mu=0.55>0\), but \(\mathbb E\operatorname{CL}(g)=-0.8\), so expected alignment is \(-0.44\). This negative case is retained in both the Phase 1 adversarial result and the Phase 2 stochastic suite. No unconditional stochastic-convergence claim is made.

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

### 7.1 Algebraic rank is not the mechanism

Exact rank is discontinuous and can be created by arbitrarily small singular values. In the included exact rational example, a rank-one \(4\times4\) outer product maps to exact rank four. Yet for 200 random log-normal factor pairs at each of sizes 4, 8, 16, and 32, median output stable rank was only about 1.00005–1.00017. The algebraic result therefore does **not** establish useful spectral diversity.

The mechanism claim can be made sharper on an isolated two-mode problem. Let a diagonal \(2\times2\) gradient have nonzero diagonal entries \(g_1,g_2\) and ratio \(q=g_1/g_2\). The two active cotransverse denominators are \(2g_2^2\) and \(2g_1^2\), so the CauchyLift direction has diagonal ratio

\[
\frac{d_1}{d_2}=q^3.
\tag{13a}
\]

Now consider a positive diagonal quadratic with arbitrary curvatures \(\lambda_1,\lambda_2\), and take an exact line search along this direction. The next gradient \(g^+\) is orthogonal to \(d\). In the nondegenerate case \(g_2^+\ne0\),

\[
q^3g_1^++g_2^+=0
\quad\Longrightarrow\quad
\boxed{q^+=\frac{g_1^+}{g_2^+}=-q^{-3}.}
\tag{13b}
\]

The curvature values cancel from this ratio law. Normalized gradient gives \(q^+=-q^{-1}\). Sign descent, a fully row/column-normalized diagonal direction, and the exact polar direction all have unit-magnitude coordinate ratios and give \(|q^+|=1\). CauchyLift therefore predicts the falsifiable log-slope

\[
\log|q^+|=-3\log|q|,
\]

versus slope \(-1\) for normalized gradient and zero for the three unit-ratio controls.

This is **mode alternation**, not a monotone deflation or acceleration theorem. Applying the idealized law twice yields \(q^{++}=q^9\): after strongly suppressing one mode, the rule can concentrate even more strongly on the other. The signature explains both the strong axis-aligned exact-line cases and the danger of scheduled steps, curvature mixing, stochasticity, and rotations. Phase 3 diagnostics should test the local log-slope and alternating concentration; failure to observe it where the two-mode approximation is accurate falsifies the mechanism, while observing it still does not prove task-level speedup.

## 8. Complexity and implementation model

For an \(m\times n\) gradient, CauchyLift needs:

- one elementwise square;
- row/column sums and linear prefix/suffix exclusion sums;
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

The radius \(\sqrt{\min(m,n)}\) is frozen for the initial experiments. It is not a tuned layer multiplier: it is the largest transpose-symmetric radius satisfying both average-fiber bounds. For a vocabulary-by-hidden embedding or head it gives squared radius equal to hidden width; for square attention matrices it gives squared radius equal to width; for expansion and contraction matrices it gives squared radius equal to the smaller dimension; and for vectors and scalars it gives unit radius. `width_suite.json` checks these identities from width 64 through 1,024 and on representative 125M-scale decoder shapes.

### 8.1 Finite-precision execution

Input gradients may be BF16, FP16, FP32, or FP64, but squares, marginal sums, exclusions, and the raw-field norm must accumulate in FP32 or higher. Max-absolute rescaling before squaring prevents overflow: all scaled squares are at most one. A naive FP32 square can overflow above approximately \(1.84\times10^{19}\), while this rescaling remains projectively exact.

Near a dominant cell, computing \(2S-r_i-c_j\) can lose the small positive complement by cancellation. A general implementation must either sum nonnegative excluded energies directly or validate the direct form and recompute a rounded invalid active denominator in FP64. The strict implementation takes the latter route. The timed direct-form path is used only for an explicitly prevalidated suite with a positive finite FP32 denominator lower bound. Only an input represented as exactly one-sparse may take the projective boundary branch. Any other zero active denominator is a diagnostic failure.

No additive epsilon is permitted. `finite_precision_suite.json` shows that even a fixed \(10^{-3}S\) epsilon measurably changes an ordinary direction. Underflowed off-dominant amplitudes are less dangerous than raw denominator condition numbers suggest because (8a) makes their projective influence cubic, but that observation justifies a boundary branch only after represented-support and rare-path checks; it does not authorize a tunable stabilizer. The machine-readable Phase 3 contract is `spec/optimizer_v0.2.json`.

Phase 3 implemented this graph as an FP64 CPU oracle, an exclusion-safe PyTorch reference, and native HIP kernels on one MI300X. Exhaustive and adversarial numerical tests pass within predeclared FP64, FP32, and BF16 tolerances; the optimizer retains zero persistent tensors and zero persistent bytes. ROCprofiler traces prove that the five custom multi-tensor kernels execute. The representative BF16 Transformer-shaped optimizer-only median is 1.0625 ms versus 0.9602 ms for fused AdamW. The 1.1065 ratio is below the preregistered maximum 1.15, so the single-MI300X Phase 3 gate is `PASS`. This is kernel engineering evidence, not neural-training evidence, and the broader second-GPU-family replication requirement remains open.

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
| \(10^2\) | rotated | 306.5 / −5.246 | 571.5 / −4.037 | 201.5 / −4.761 | 323 / −3.088 | **78 / −5.196** |
| \(10^4\) | axis aligned | 600 / −3.150 | 600 / **−5.265** | 317 / **−5.922** | 600 / −4.018 | **208 / −2.964** |
| \(10^4\) | rotated | 600 / −3.356 | 600 / −2.998 | 600 / −3.307 | 600 / **−4.640** | 600 / −3.078 |

At condition \(10^2\), CauchyLift is strong under both probes. At condition \(10^4\), it retains good axis-aligned exact-line direction quality but performs poorly with the scheduled step, and every rotated exact-line run hits the cap. This is evidence of step sensitivity and basis dependence, not evidence of SOAP/Muon-like convergence.

### 9.3 Rejected-candidate checks

The exterior/cofactor candidate's condition map matches a Halley triple-angle identity to numerical precision. The cross-ratio plaquette dual fails 198 of 200 exact-line probes at the 1,000-step cap, versus 46 for sign. These results are kept so that rejected ideas are not later rediscovered and marketed under new names.

### 9.4 Phase 2 boundary, noise, shape, and precision suites

The expanded standard-library-only suite records seed `20260828` and includes:

- exhaustive enumeration of 1,554 nonzero matrices over \(\{-1,0,1\}\) in every shape from \(1\times1\) through \(3\times2\) listed in the artifact;
- 10,000 additional property cases over all shapes from \(1\times1\) through \(6\times6\), with zeros and log-uniform dynamic range;
- direct checks of the boundary modulus, \(\Theta(1/\tau)\) denominator growth, the interior Lipschitz bound, and discontinuity at zero;
- exact finite-distribution checks of (10b)--(10f), including a benign positive-margin distribution and the expected-ascent counterexample;
- the exact cubic two-mode recurrence against normalized-gradient, sign, row/column-normalized, and polar controls;
- width-transfer identities, zero/boundary totality for scalar/vector/matrix shapes, BF16/FP16 input quantization, a high-precision decimal oracle, overflow, underflow, cancellation, and epsilon-collision cases.

Every positive gate passed. The expected-ascent noise distribution, zero-discontinuity sequence, subtractive-cancellation case, raw denominator growth, two-step \(q^9\) concentration, hard rotated quadratics, and near-one stable-rank results remain explicit negatives.

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
2. **Limited wall-clock evidence.** The MI300X native path passes the isolated Phase 3 optimizer-step gate at 1.1065× fused AdamW, but no end-to-end training wall time or second GPU family has been measured.
3. **Only conditional stochastic theory.** The measurable SNR and high-probability conditions are sufficient, not known necessary, and may fail in language-model training.
4. **Basis dependence.** Only permutations and transpose are symmetries; arbitrary rotations are not.
5. **Step sensitivity.** The condition-\(10^4\) scheduled result is a direct warning.
6. **Algebraic versus numerical rank.** Generic full exact rank can carry negligible extra singular mass.
7. **Boundary implementation is validated on only one GPU.** The exact boundary and FP64 rare path pass the Phase 3 suite, but independent hardware replication remains absent.
8. **Radius scaling is derived but not empirically optimal.** \(\sqrt{\min(m,n)}\) is frozen by symmetric average-fiber constraints; those constraints are not a theorem about best neural-training scale.
9. **Finite novelty search.** No search can prove that an equivalent unpublished or differently named method does not exist.
10. **No acceleration theorem.** The stationarity result matches normalized first-order order, and the exact mode result predicts aggressive alternation as well as deflation.
11. **Sparse concentration risk.** Embedding gradients with few active rows can concentrate a fixed layer radius on that support; no fallback is allowed, so this is a required diagnostic and possible kill result.
12. **Sensitivity constants are loose.** The regional Lipschitz constant proves control but is far larger than observed local ratios and is not a useful tuning formula.

## 12. Phase 3 outcome and next gate

Phase 3 implemented only `spec/optimizer_v0.2.json`, with no momentum, epsilon, or rescue mixture. Its correctness, safety, state, memory, and native-execution checks pass, but its optimizer-step speed gate fails. Phase 4 and the training study are therefore not authorized. Any future engineering revision must preserve the exact map, rerun the full oracle/HIP suite, clear the 1.15× limit, and obtain independent replication on a second modern GPU family.

The hypothesis should be rejected if it requires a conventional optimizer component to become stable or competitive, exceeds AdamW optimizer-step time by more than 15% after fusion, or fails the predeclared tokens-to-target criterion on most workloads. The full protocol is in `research/future_experiment_protocol.md`.

## 13. Conclusion

CauchyLift proposes a mathematical direction absent from the searched optimizer families: projective dualization by cotransverse row-column energy. The primitive is simple enough for linear reduction-only execution yet structured enough to yield a uniform-angle theorem, a stable projective boundary, conditional noisy-gradient descent, an exact Cauchy-kernel law, and a cubic two-mode recurrence. Those are genuine theoretical results. They do not establish the desired empirical outcome. The expected-ascent noise example, two-step concentration, hard rotated quadratics, sparse-shape risk, and stable-rank negatives make implementation diagnostics decisive.

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
