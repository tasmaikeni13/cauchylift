# CauchyLift

**CauchyLift** is a curvature-adaptive matrix optimizer designed for large-scale deep learning and foundation model pretraining. It combines **historical directional momentum filtering**, an **Additive Fiber RMS Cauchy Operator**, and **decoupled weight decay** to deliver second-order-like curvature adaptation with strictly linear computational complexity and minimal memory overhead.

---

## Key Highlights

* **50% Less Optimizer Memory than AdamW:** Tracks only a single momentum state tensor per parameter (4 bytes/param in FP32, or 2 bytes/param in BF16), eliminating the second moment tensor ($V_t$) and reducing memory pressure during large-model pretraining.
* **Linear $O(N^2)$ Computational Complexity:** Replaces expensive $O(N^3)$ matrix inversions, SVD, and iterative polar decompositions (such as Newton–Schulz iterations in Muon) with parallel row and column RMS fiber reductions.
* **Sub-Millisecond Native TPU/XLA Kernels:** Fused multi-tensor operations lowered to Google Cloud TPU v6e (Trillium) HLO execute in $<0.8$ ms across full Transformer parameter groups—over $15\times$ faster than iterative matrix orthogonalizers.
* **Mathematically Proven Stability:** Rigorously proven degree-0 scale invariance ($U(\alpha M) = U(M)$), coordinate magnitude bounds ($|Z_{ij}| \le \min(\sqrt{n}, \sqrt{m})$), and strict positive descent alignment ($\langle M, U(M) \rangle > 0$).
* **Empirically Validated Pretraining:** Pretrained decoder-only Transformers on FineWeb-Edu up to 3B tokens per run across Google Cloud TPU v6e clusters, achieving smooth monotonic convergence with tight cross-seed variance and zero loss spikes.

---

## The CauchyLift Algorithmic Formulation

For a parameter matrix $W \in \mathbb{R}^{m \times n}$ and stochastic gradient $G_t$:

1. **Directional Velocity Filtering (Historical Momentum):**
   $$M_t = \beta M_{t-1} + (1 - \beta) G_t \quad (\beta = 0.95)$$
   Attenuates high-frequency stochastic minibatch variance by $\sim 39\times$, filtering transverse noise across loss canyon walls.

2. **Additive Fiber RMS Cauchy Lifting:**
   $$D_{ij}(M_t) = \text{RMS}(M_{t, i, :}) + \text{RMS}(M_{t, :, j}) = \sqrt{\frac{1}{n} \sum_{k=1}^n M_{ik}^2} + \sqrt{\frac{1}{m} \sum_{l=1}^m M_{lj}^2}$$
   $$Z_{ij}(M_t) = \frac{M_{t, ij}}{D_{ij}(M_t)}$$

3. **Longest-Fiber Radius Normalization:**
   $$U(M_t) = \sqrt{\max(m, n)} \frac{Z(M_t)}{\|Z(M_t)\|_F}$$
   Ensures uniform step authority across rectangular matrices and vocabulary embeddings.

4. **Decoupled Weight Decay & Parameter Update:**
   $$W_{t+1} = W_t (1 - \eta_t \lambda) - \eta_t U(M_t)$$
   Actively prevents parameter norm runaway in scale-invariant Pre-RMSNorm Transformer architectures.

---

## Quickstart

### Installation
```bash
git clone https://github.com/tasmaikeni13/cauchylift.git
cd cauchylift
pip install -r requirements/tpu-v6e.txt
pip install -e .
```

### PyTorch / Torch-XLA Usage
```python
import torch
import torch_xla.core.xla_model as xm
from cauchylift import CauchyLift, get_tpu_device, sync_tpu

device = get_tpu_device()
model = MyTransformer().to(device=device, dtype=torch.bfloat16)

optimizer = CauchyLift(
    model.parameters(),
    lr=1e-3,            # Learning rate
    momentum=0.95,      # Historical momentum factor
    weight_decay=0.01,  # Decoupled weight decay
    backend="auto",     # "auto" selects native XLA on TPU, reference on CPU
)

# Training loop
for tokens, targets in dataloader:
    optimizer.zero_grad()
    tokens, targets = tokens.to(device), targets.to(device)
    loss = model(tokens, targets)
    loss.backward()
    optimizer.step()
    sync_tpu()
```

---

## Verification & Testing

### 1. Run Full Smoke Test
Validates imports, reference math, native TPU/XLA fused kernels, memory accounting, and a full Transformer step on TPU:
```bash
python scripts/smoke_test.py
```

### 2. Run PyTest Suite
```bash
pytest -v
```

---

## Repository Structure

* [`cauchylift/`](cauchylift/) — Core Python package containing `CauchyLift` optimizer, reference implementation, and TPU/XLA accelerator.
* [`phases/`](phases/) — Nine autonomous research phases guiding theory, verification, scaling, and publication.
* [`scripts/`](scripts/) — Standalone smoke test (`smoke_test.py`) and production pretraining script (`train_transformer.py`).
* [`tests/`](tests/) — Complete test suite covering attention, models, reference math, and Google Cloud TPU tests.

---

## License

MIT License. See [LICENSE](LICENSE) for details.
