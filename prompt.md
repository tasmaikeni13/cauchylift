# MISSION: Production CauchyLift Implementation, Full 3B-Token Sweep, and Repository Deployment

You are the lead systems and ML engineer on CauchyLift. You have access to a 16-chip Google Cloud TPU v4 slice. 

Your mission is to build the production-grade, canonical version of CauchyLift, update `paper.md`, prepare the real dataset, execute an exhaustive multi-seed pretraining sweep to 100% completion across all steps (zero mock runs, zero early breaks), and push all code, results, and artifacts to GitHub.

---

### 1. Production Optimizer Implementation
Implement the canonical, single-class **CauchyLift** optimizer with native Torch-XLA / PJRT acceleration for Google Cloud TPU v4:
* **Canonical Parameter Routing:**
  - **2D Hidden Linear Matrices:** Update via the core CauchyLift formulation (momentum low-pass filter, Additive Fiber RMS curvature denominator, longest-fiber Frobenius projection, decoupled weight decay).
  - **1D Parameters & Embeddings:** Automatically route all 1D parameters (RMSNorm/LayerNorm scales, biases) and token lookup/head matrices to coordinate-wise AdamW updates. This is the default, native behavior of CauchyLift—do not treat or label it as a "hybrid."
* **Hardware & Systems Optimization:**
  - Maximize hardware execution across the 16 TPU v4 chips using fused XLA graphs and multi-tensor execution.
  - Maintain FP32 accumulation for all reductions (fiber RMS and Frobenius sums) inside TPU vector registers to prevent underflow, while executing matrix multiplications in native BF16.
  - Ensure zero host-device sync overhead inside the training step loop.

---

### 2. Theoretical Documentation
Update `paper.md` to reflect the canonical design:
* Document the mathematical formulation and theoretical justification for the parameter decomposition: explain why dense 2D linear transformation operators require matrix-geometric fiber curvature and Frobenius sphere projection, while 1D calibration scalars and sparse Zipf-distributed token embedding tables require coordinate-wise adaptive variance.
* Detail the unified multi-tensor execution pipeline and its latency/memory profile across Google Cloud TPU v4 MXUs and VPUs.

---

### 3. Dataset Acquisition & Verification
* Download, tokenize, and pre-pack the real **FineWeb-Edu** dataset into contiguous chunks of 2,048 tokens.
* Verify locally that the binary cache contains at least **3.05 billion valid tokens** (3B train + validation split) before launching any training. 
* Do not proceed until total token counts on disk are strictly verified.

---

### 4. Full 125M Hyperparameter Sweep (3B Tokens per Run)
Execute a distributed pretraining sweep on a 125M-parameter Transformer across your 16 TPU v4 chips (262,144 tokens per global step, ~11,444 steps per run):
* **Runs & Search Space:**
  - **CauchyLift:** Sweep base matrix learning rates across `[0.0025, 0.0050, 0.0100]` with seeds `[42, 43, 44]` (9 runs total).
  - **AdamW Baseline:** Run at its standard `0.0006` learning rate across seeds `[42, 43, 44]` (3 runs total).
* **Execution Mandate (Anti-Truncation):**
  - Every single run must train for the full 3B tokens (~11,444 steps). Absolutely zero simulated runs, no early loop breaks, and no mock steps.
  - Ensure explicit TPU memory cleanup and synchronization between runs so subsequent jobs do not crash with HBM allocation errors.
  - Evaluate and log validation loss periodically (every 500 steps) and save run summaries (validation loss, final training loss, tokens/sec, MFU).

---

### 5. Aggregation & GitHub Deployment
* Synthesize all completed runs into a final summary report ranking configurations by **Mean Validation Loss** and **Cross-Seed Standard Deviation** alongside throughput metrics.
* Commit all working code, benchmark scripts, result logs, and the updated `paper.md`.
* Push the repository directly to GitHub.
