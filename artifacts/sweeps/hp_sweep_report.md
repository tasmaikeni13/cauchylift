# Hyperparameter Tuning Sweep Report (125M Transformer on FineWeb-Edu)

## Executive Summary

This report details the hyperparameter sweep conducted across a 16-chip Google Cloud TPU v4 slice 
(v4-32, 4 worker hosts) on real FineWeb-Edu pre-packed tokens (200M tokens per calibration arm).
Both **CauchyLift** and **AdamW** were systematically tuned across candidate learning rates with 3 random seeds 
(42, 43, 44) to guarantee a fair, statistically sound comparison for the subsequent 3B-token pretraining runs.

### Optimal Hyperparameters Found

- **Optimal CauchyLift Matrix LR**: `0.0025` (Mean Val Loss: **3.6687 ± 0.0059**)
- **Optimal AdamW LR**: `0.0020` (Mean Val Loss: **3.4638 ± 0.0058**)

## Full Sweep Ranking Table

| Optimizer | Learning Rate | Completed Seeds | Mean Val Loss | Val Loss Std | Mean Perplexity | Throughput (tok/s) | MFU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADAMW** | `0.0020` | 3/3 | **3.4638** | 0.0058 | 31.94 | 1,073,652 | 18.1% |
| **ADAMW** | `0.0010` | 3/3 | **3.5335** | 0.0027 | 34.24 | 1,068,209 | 18.0% |
| **ADAMW** | `0.0006` | 3/3 | **3.6779** | 0.0068 | 39.56 | 1,062,903 | 17.9% |
| **ADAMW** | `0.0003` | 3/3 | **4.1098** | 0.0039 | 60.94 | 1,022,709 | 17.2% |
| **CAUCHYLIFT** | `0.0025` | 3/3 | **3.6687** | 0.0059 | 39.20 | 1,062,827 | 17.9% |
| **CAUCHYLIFT** | `0.0050` | 3/3 | **3.7598** | 0.0434 | 42.97 | 1,067,436 | 18.0% |
| **CAUCHYLIFT** | `0.0010` | 3/3 | **3.8412** | 0.0055 | 46.58 | 1,029,412 | 17.3% |
| **CAUCHYLIFT** | `0.0100` | 3/3 | **6.2392** | 0.1365 | 515.65 | 1,033,017 | 17.4% |

## Hardware and Cluster Setup
- **Cluster**: Google Cloud TPU v4-32 (16 chips, 32 TensorCores across 4 hosts in a 2x2x4 3D Torus optical mesh)
- **Software**: PyTorch 2.5 + Torch-XLA PJRT, Native BF16 MXU matrix ops with FP32 VPU register reductions
- **Dataset**: Real FineWeb-Edu (3.05B verified tokens on disk)
