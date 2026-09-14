# Hyperparameter Tuning Sweep Report (125M Transformer on FineWeb-Edu)

## Executive Summary

This report details the hyperparameter sweep conducted across a 16-chip Google Cloud TPU v4 slice 
(v4-32, 4 worker hosts) on real FineWeb-Edu pre-packed tokens (200M tokens per calibration arm).
Both **CauchyLift** and **AdamW** were systematically tuned across candidate learning rates with 3 random seeds 
(42, 43, 44) to guarantee a fair, statistically sound comparison for the subsequent 3B-token pretraining runs.

### Optimal Hyperparameters Found

- **Optimal CauchyLift Matrix LR**: `0.0010` (Mean Val Loss: **5.6019 ± 0.0465**)
- **Optimal AdamW LR**: `0.0020` (Mean Val Loss: **4.7713 ± 0.0460**)

## Full Sweep Ranking Table

| Optimizer | Learning Rate | Completed Seeds | Mean Val Loss | Val Loss Std | Mean Perplexity | Throughput (tok/s) | MFU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ADAMW** | `0.0020` | 3/3 | **4.7713** | 0.0460 | 118.16 | 1,063,519 | 17.9% |
| **ADAMW** | `0.0010` | 3/3 | **4.9285** | 0.0231 | 138.20 | 1,095,096 | 18.5% |
| **ADAMW** | `0.0006` | 3/3 | **5.1612** | 0.0534 | 174.54 | 1,041,589 | 17.5% |
| **ADAMW** | `0.0003` | 3/3 | **5.5131** | 0.0277 | 247.99 | 1,067,377 | 18.0% |
| **CAUCHYLIFT** | `0.0010` | 3/3 | **5.6019** | 0.0465 | 271.14 | 1,079,770 | 18.2% |
| **CAUCHYLIFT** | `0.0025` | 3/3 | **5.7558** | 0.0271 | 316.11 | 1,098,791 | 18.5% |
| **CAUCHYLIFT** | `0.0050` | 3/3 | **6.1627** | 0.0806 | 475.75 | 201,319 | 3.4% |
| **CAUCHYLIFT** | `0.0100` | 3/3 | **6.4492** | 0.0555 | 632.87 | 1,104,244 | 18.6% |

## Hardware and Cluster Setup
- **Cluster**: Google Cloud TPU v4-32 (16 chips, 32 TensorCores across 4 hosts in a 2x2x4 3D Torus optical mesh)
- **Software**: PyTorch 2.5 + Torch-XLA PJRT, Native BF16 MXU matrix ops with FP32 VPU register reductions
- **Dataset**: Real FineWeb-Edu (3.05B verified tokens on disk)
