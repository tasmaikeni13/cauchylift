# Closest-work matrix

| Family | State | Per-step structure | Defining operation | Why CauchyLift is not that operation | Residual collision risk |
|---|---:|---|---|---|---|
| Adam/AdamW | Two full tensors | Pointwise | Historical first moment divided by RMS history | No history, no moment, and weights depend on other rows and columns | Any sign-preserving rational map can be called diagonal adaptivity in a broad sense |
| Adafactor | Row/column histories | Reductions + pointwise | Factorized historical second moment | Uses instantaneous *complements* \((S-r_i)+(S-c_j)\), not a product approximation to a moment matrix | Both use row and column reductions |
| Shampoo | Matrix histories | Matrix products and inverse roots | Kronecker covariance preconditioner | No covariance, inverse root, or left/right matrix action | Both are matrix-shaped first-order methods |
| SOAP | Matrix histories plus element state | Eigenbasis updates | Adam in a changing Shampoo basis | No basis, Adam state, or eigendecomposition | Same high-level target of cheap matrix awareness |
| Muon | Momentum tensor | Matrix multiplications | Approximate polar factor / matrix sign | CauchyLift is entry-coupled and generally changes singular vectors; no matrix polynomial | Both use fixed-norm matrix updates |
| Scion | Typically stateless | Depends on norm LMO | Linear minimizer over a norm ball | The cotransverse rational field is not an LMO solution supplied by a fixed norm | A gradient-dependent gauge could possibly represent it after the fact |
| SinkGD / multi-normalization | Stateless | Alternating row/column reductions | Balances direct row and column norms to a fixed point | One pass over energy *outside* the row/column; it sharpens concentrated intersections rather than balancing direct marginals | Closest implementation pattern |
| SWAN | Stateless | Row reduction + whitening matmuls | Row normalization then whitening | No sequential normalization/whitening modules | Both transform instantaneous matrices |
| RACS/MOGA/row-column rules | Varies | Row/column reductions | Direct marginal scaling or structured Fisher/operator-norm rule | Reciprocal of the sum of marginal complements; not a direct marginal inverse and not separable into row factor × column factor | Closest conceptual family; formula search must be repeated |
| ARO | Rotation state | QR + base optimizer | Learns/tracks a rotation, then applies another map | No rotation and no base optimizer | None beyond matrix awareness |
| Low-rank methods (GaLore/Alice/Dion) | Subspace state | Projection/factor operations | Restrict or transport updates in low-dimensional subspaces | No subspace selection or truncation; exact rank can increase | “Rank” language could create superficial confusion |
| Gauss–Southwell | Usually none | Selection/top-k | Update largest-gradient coordinates/blocks | CauchyLift is dense and smooth away from its projective boundary | Its concentration behavior resembles a soft greedy rule |
| Smoothed nuclear-norm online methods | Potential-dependent | Matrix-potential gradients | Adaptive online matrix learning via smoothed spectral/nuclear-norm potentials | CauchyLift has no cumulative online state or spectral potential evaluation | Both provide matrix-valued geometry; formula and operation graphs differ |

## Non-compositionality test

The final rule contains one tensor-valued operation: construct the cotransverse field and take its projective dual. Scalar radius assignment is unavoidable for any direction method and is not counted as a second optimizer. There is no momentum, adaptive moment, whitening, rotation, low-rank projector, sign branch, clipping branch, or fallback optimizer.

The strongest objection is that the update can be written entrywise as \(D_{ij}=p_{ij}(G)G_{ij}\). That observation is true but not discriminative: every sign-preserving direction has such a representation. The novelty question is therefore about the specific coupled field \(p_{ij}(G)=1/[(S-r_i)+(S-c_j)]\), its projective boundary, and its derived theorems.

The 2026-08-28 operational decomposition audit also tried the following rewrites: one left diagonal scaling followed by one right diagonal scaling, a finite row/column balancing pass, a fixed-norm steepest-descent/LMO step, and a polar/whitening map. The reciprocal additive denominator is generically nonseparable, one evaluation has no balancing fixed point, its direction is not the optimizer of a fixed gradient-independent norm ball, and it changes singular vectors without computing a spectral map. No reusable known-module decomposition was found within these tested families.
