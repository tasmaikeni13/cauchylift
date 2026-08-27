# Literature search log

Search date: 2026-08-27 UTC. Search engines: arXiv, OpenReview, PMLR, and general web search used only to locate primary papers. The substantive evidence set uses original papers or official proceedings pages.

## Query lattice

### Established optimizer mechanisms

- `Shampoo tensor preconditioner Kronecker inverse root primary paper`
- `SOAP optimizer Adam Shampoo eigenbasis`
- `Muon Newton Schulz polar optimizer convergence`
- `matrix whitening optimizer deep learning variance adaptation`
- `structured Fisher optimizer row column low rank LLM`
- `matrix operator norm row column normalization optimizer`
- `norm constrained LMO optimizer Scion`
- `SinkGD gradient multi-normalization stateless`
- `SWAN row normalization whitening optimizer`
- `adaptive rotation optimizer ARO matrix optimization`
- `block Muon periodic orthogonalization`
- `innovation augmented polar optimizer PRISM`

### Direct-formula collision search

- `optimizer gradient "sum of squares" minus coordinate squared denominator`
- `optimizer gradient "row energy" "column energy" complement`
- `matrix optimizer "outside the row" gradient energy`
- `optimizer "complement energy" gradient matrix`
- `optimizer "leave-one-out" gradient norm`
- `optimizer "Cauchy kernel" gradient update`
- `optimizer "CauchyLift"`
- `optimizer "cotransverse energy"`
- `optimizer "rank-lifting" gradient`
- `optimizer "increase the rank" gradient update matrix`

### Rejected-branch collision search

- `cofactor optimizer neural network gradient`
- `adjugate gradient optimizer matrix`
- `exterior algebra optimizer deep learning`
- `geodesic extrapolation gradient direction sphere optimizer`
- `rotor gradient optimizer`
- `cross-ratio optimizer gradient matrix`
- `plaquette gradient optimizer neural network`
- `gauge invariant optimizer matrix gradient`
- `gradient 2x2 minors denoising optimizer`
- `gradient cycle code sign matrix optimizer`

### Recency sweep

- `site:arxiv.org 2026 matrix optimizer deep learning`
- `site:openreview.net ICLR 2026 matrix optimizer`
- `site:arxiv.org 2026 gradient transformation optimizer neural network`
- citations and related-work graphs from ARO, the matrix-operator-norm paper, SOAP, Muon theory, SinkGD, and RACS/Alice.
- `symmetry compatible optimizer bi-orthogonal spectral matrix update`

## Saturation record

The search was expanded until new results repeatedly fell into already represented families:

1. historical moments and diagonal adaptivity;
2. covariance/Fisher/Kronecker preconditioning;
3. spectral, polar, whitening, or matrix-sign transforms;
4. norm-ball/LMO geometry;
5. row/column normalization or matrix scaling;
6. rotations or basis tracking;
7. low-rank projection/sketching;
8. sign, momentum, or temporal extrapolation;
9. wrappers that rescale another optimizer.

No searched source used the exact field
\(G_{ij}/[(S-r_i)+(S-c_j)]\) followed by projective normalization. This is a finite, query-dependent negative result, not proof of universal novelty.

## Name collision search

Queries for `CauchyLift optimizer`, `Cauchy Lift gradient optimizer`, and close variants found no machine-learning optimizer with that name as of the search date. “RankLift” was rejected as a name because it is used by unrelated products.

## Update policy

Before any public empirical claim, repeat:

- forward citation searches on all closest papers;
- exact formula searches with algebraically rearranged denominators;
- GitHub code search for `2 * sum(g*g) - row_sum - col_sum`-style expressions;
- a fresh arXiv/OpenReview sweep from 2026-08-28 onward.
