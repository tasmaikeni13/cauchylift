# Reproducibility Guide for CauchyLift

This repository provides end-to-end reproducibility for the mathematical proofs, native ROCm/HIP kernels, and empirical pretraining benchmarks of the CauchyLift optimizer.

---

## 1. Environment & Hardware Specification

* **Operating System:** Linux (Ubuntu 22.04 LTS or compatible)
* **Target Hardware:** AMD Instinct MI300X (192 GB HBM3, 304 CUs, `gfx942`)
* **Software Stack:**
  * ROCm 10.0.0
  * Python 3.12.3
  * PyTorch 2.13.0+rocm10.0.0 (`torch[device-gfx942]`)
  * Ninja 1.13.0, PyTest 8.4.2

Install required dependencies:
```bash
pip install -r requirements/rocm10-mi300x.txt
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

Verify that the native ROCm/HIP kernels, memory tracking, and Transformer FlashAttention steps execute cleanly on the MI300X GPU:
```bash
python scripts/smoke_test.py
```

---

## 4. Complete Unit & Regression Test Suite

Execute the entire test suite across all modules:
```bash
pytest -v
```
All 67 tests across model architecture, attention, reference implementations, and native ROCm kernels should pass with 100% success.

---

## 5. 125M Transformer Pretraining on FineWeb-Edu

To reproduce the 125M pretraining run on FineWeb-Edu tokens:
```bash
python scripts/train_transformer.py \
  --data_train data/fineweb_edu/train_tokens_350m.bin \
  --data_val data/fineweb_edu/val_tokens.bin \
  --total_tokens 100000000 \
  --seq_len 4096 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 0.001 \
  --momentum 0.95 \
  --weight_decay 0.01 \
  --compile \
  --output_dir runs/cauchylift_125m_repro
```
