# Reproducibility Guide for CauchyLift

This repository provides end-to-end reproducibility for the mathematical proofs, native Google Cloud TPU / XLA kernels, and empirical pretraining benchmarks of the CauchyLift optimizer.

---

## 1. Environment & Hardware Specification

* **Operating System:** Linux (Ubuntu 22.04 LTS or compatible)
* **Target Hardware:** Google Cloud TPU v6e (Trillium) (8 TPU chips, `v6e-8`, 2x4 topology)
* **Software Stack:**
  * Torch-XLA 2.9.0 with libtpu
  * Python 3.10+
  * PyTorch 2.9.0+cpu
  * PyTest 9.1+

Install required dependencies:
```bash
pip install -r requirements/tpu-v6e.txt
pip install -e .
```

---

## 2. Mathematical Audit Reproduction

Run the standalone standard-library mathematical audit verifying Theorem 1 (scale invariance), Theorem 2 (coordinate bounds), Theorem 3 (descent alignment), and radius normalization:
```bash
python3 analysis/run_mathematical_audit.py
```
Expected output:
```
Testing Theorem 1: Degree-0 Scale Invariance... [PASS]
Testing Theorem 2: Coordinate Magnitude Bounds... [PASS]
Testing Theorem 3: Strict Descent Alignment... [PASS]
Testing Radius Normalization... [PASS]
ALL MATHEMATICAL THEOREMS VERIFIED SUCCESSFULLY!
```

---

## 3. Full-Stack Smoke Test Reproduction

Verify that the native Google Cloud TPU / XLA kernels, memory tracking, and Transformer steps execute cleanly on the TPU v6e:
```bash
python scripts/smoke_test.py
```

---

## 4. Complete Unit & Regression Test Suite

Execute the entire test suite across all modules:
```bash
pytest -v
```
All unit tests across model architecture, attention, reference implementations, and Google Cloud TPU kernels should pass with 100% success.

---

## 5. Distributed Multi-Core & MFU Scaling Reproduction (Phase 6)

### A. Verify 8-Chip Multi-Core Orchestration & Zero Drift:
```bash
python scripts/verify_8x_tpu_orchestration.py
```
Expected output: Bitwise identical parameters across all 8 ranks (`max parameter drift = 0.000000e+00`).

### B. Run Throughput & MFU Scaling Profiler:
```bash
python scripts/benchmark_mfu_scaling.py --mode all
```
Expected output: Single-chip ~103k tok/s (21.8% MFU), 8-chip distributed ~789k tok/s (20.8% MFU, 95.1% parallel efficiency).

### C. Run Hyperparameter Pilot Sweeps:
```bash
python scripts/run_phase6_pilot_sweep.py
```

---

## 6. Confirmatory 3B-Token Pretraining (Phase 7 Protocols)

Pretraining configurations are frozen in `experiments/protocols/protocol_125m_fineweb.json` and `experiments/protocols/protocol_350m_fineweb.json`.

To run CauchyLift on 125M (3B tokens):
```bash
python scripts/train_transformer.py \
  --data_train data/fineweb_edu/train_tokens_350m.bin \
  --data_val data/fineweb_edu/val_tokens.bin \
  --total_tokens 3000000000 \
  --seq_len 2048 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 0.005 \
  --momentum 0.95 \
  --weight_decay 0.01 \
  --optimizer cauchylift \
  --seed 42 \
  --output_dir runs/cauchylift_125m_seed42
```

To run AdamW on 125M (3B tokens):
```bash
python scripts/train_transformer.py \
  --data_train data/fineweb_edu/train_tokens_350m.bin \
  --data_val data/fineweb_edu/val_tokens.bin \
  --total_tokens 3000000000 \
  --seq_len 2048 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 0.0006 \
  --momentum 0.95 \
  --weight_decay 0.01 \
  --optimizer adamw \
  --seed 42 \
  --output_dir runs/adamw_125m_seed42
```
