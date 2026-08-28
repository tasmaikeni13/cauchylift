# Literature search log

Search dates: 2026-08-27 and 2026-08-28 UTC. Search engines: arXiv, OpenReview, PMLR, Google Patents, public GitHub code search, and general web search used only to locate primary papers or public implementations. The substantive evidence set uses original papers, official proceedings pages, patent records, or source repositories.

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

### Phase 1 formula, implementation, thesis, and patent sweep — 2026-08-28

- `"2S - r_i - c_j" optimizer gradient matrix`
- `"row energy" "column energy" optimizer gradient complement`
- `"outside the row" "outside the column" gradient matrix`
- `optimizer gradient "sum of squares" minus row sum minus column sum`
- `site:github.com optimizer "row_sum" "col_sum" gradient`
- `site:github.com "2 * total" row column optimizer gradient`
- `site:github.com "sum(g * g)" row column optimizer`
- `site:github.com CauchyLift optimizer`
- `site:arxiv.org optimizer "leave-one-out" gradient normalization`
- `site:arxiv.org optimizer complement row column gradient matrix`
- `site:arxiv.org optimizer "Cauchy kernel" gradient descent matrix`
- `site:arxiv.org rational gradient field optimization preconditioner`
- `patent optimizer gradient row column sum squares complement energy`
- `site:arxiv.org/abs/2608 optimizer matrix gradient stateless row column`
- `site:arxiv.org/abs/2607 matrix optimizer deep learning gradient normalization`
- `site:openreview.net 2026 optimizer matrix stateless gradient`
- `site:proceedings.mlr.press optimizer row column gradient normalization matrix`

The GitHub formula queries returned unrelated image-processing reductions and Sinkhorn implementations, not an optimizer computing \(G_{ij}/(2S-r_i-c_j)\). The patent query returned low-rank complement-space compensation with historical state and mixtures, not the entrywise cotransverse field. The thesis and leave-one-out queries concerned cross-validation or matrix subspaces. These are negative search results within the recorded query scope, not proof of absence.

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

No source searched through 2026-08-28 used the exact field
\(G_{ij}/[(S-r_i)+(S-c_j)]\) followed by projective normalization. This is a finite, query-dependent negative result, not proof of universal novelty.

## Name collision search

Queries for `CauchyLift optimizer`, `Cauchy Lift gradient optimizer`, and close variants found no machine-learning optimizer with that name through 2026-08-28. “RankLift” was rejected as a name because it is used by unrelated products.

## Update policy

Before any public empirical claim, repeat:

- forward citation searches on all closest papers;
- exact formula searches with algebraically rearranged denominators;
- GitHub code search for `2 * sum(g*g) - row_sum - col_sum`-style expressions;
- a fresh arXiv/OpenReview sweep from 2026-08-28 onward.
