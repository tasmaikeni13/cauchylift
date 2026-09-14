# Hyperparameter Sweeps across 16x Google Cloud TPU v4 (v4-32 slice)

## Executive Summary

This document summarizes the empirical hyperparameter sweeps on **FineWeb-Edu** evaluated
across all **16 Google Cloud TPU v4 chips** in a **2x2x4 3D Torus mesh** for:
1. **125M Decoder Transformer** (preregistered for 3,000,000,000 token budget)
2. **350M Decoder Transformer** (preregistered for 7,000,000,000 token budget)

Comparing **CauchyLift** and **AdamW** optimizers under identical model seeds (evaluated across seeds 42, 43, and 44) and disjoint data streams.

### Optimal Hyperparameters by Scale and Optimizer

| Model Scale | Optimizer | Optimal LR | Momentum | Weight Decay | Final Loss (3-Seed Mean ± Std) | Loss Drop | Latency | Cluster Throughput | MFU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **125M** | **CauchyLift** | **0.010** | 0.95 | 0.01 | **7.6541** ± 0.0417 | 3.2942 | 145.3 ms | 902,373 tok/s | 15.2% |
| **125M** | **AdamW** | **0.002** | 0.95 | 0.01 | **7.5601** ± 0.0602 | 3.3883 | 181.5 ms | 722,148 tok/s | 12.2% |
| **350M** | **CauchyLift** | **0.003** | 0.95 | 0.01 | **7.6940** ± 0.0360 | 3.3313 | 309.3 ms | 211,851 tok/s | 10.4% |
| **350M** | **AdamW** | **0.0002** | 0.95 | 0.01 | **7.6615** ± 0.0338 | 3.3638 | 364.3 ms | 179,914 tok/s | 8.8% |

## Detailed Arm Trajectories

### 125M Transformer Sweep Results

#### CAUCHYLIFT (6 arms)

| Arm | Configuration | LR | Momentum | WD | Init Loss | Final Loss (3-Seed Mean ± Std) | Loss Drop | Latency | Throughput | MFU |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | CauchyLift LR 0.0005 | 0.0005 | 0.95 | 0.01 | 10.9484 | **8.4270** ± 0.0113 | 2.5214 | 187.6 ms | 698,761 tok/s | 11.8% |
| 2 | CauchyLift LR 0.0010 | 0.001 | 0.95 | 0.01 | 10.9484 | **7.9152** ± 0.0281 | 3.0332 | 144.7 ms | 905,964 tok/s | 15.3% |
| 3 | CauchyLift LR 0.0020 | 0.002 | 0.95 | 0.01 | 10.9483 | **7.7020** ± 0.0335 | 3.2463 | 139.8 ms | 937,792 tok/s | 15.8% |
| 4 | CauchyLift LR 0.0050 | 0.005 | 0.95 | 0.01 | 10.9484 | **7.6623** ± 0.0250 | 3.2861 | 150.5 ms | 871,039 tok/s | 14.7% |
| 5 | CauchyLift LR 0.0100 | 0.01 | 0.95 | 0.01 | 10.9483 | **7.6541** ± 0.0417 | 3.2942 | 145.3 ms | 902,373 tok/s | 15.2% |
| 6 | CauchyLift LR 0.0150 | 0.015 | 0.95 | 0.01 | 10.9483 | **8.2122** ± 0.0723 | 2.7361 | 143.2 ms | 914,995 tok/s | 15.4% |

#### ADAMW (5 arms)

| Arm | Configuration | LR | Momentum | WD | Init Loss | Final Loss (3-Seed Mean ± Std) | Loss Drop | Latency | Throughput | MFU |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | AdamW LR 0.0001 | 0.0001 | 0.95 | 0.01 | 10.9484 | **8.2108** ± 0.0115 | 2.7376 | 267.4 ms | 490,086 tok/s | 8.3% |
| 2 | AdamW LR 0.0003 | 0.0003 | 0.95 | 0.01 | 10.9484 | **7.6468** ± 0.0372 | 3.3016 | 174.6 ms | 750,507 tok/s | 12.6% |
| 3 | AdamW LR 0.0006 | 0.0006 | 0.95 | 0.01 | 10.9484 | **7.6348** ± 0.0362 | 3.3136 | 181.0 ms | 724,225 tok/s | 12.2% |
| 4 | AdamW LR 0.0010 | 0.001 | 0.95 | 0.01 | 10.9484 | **7.6306** ± 0.0361 | 3.3178 | 182.2 ms | 719,408 tok/s | 12.1% |
| 5 | AdamW LR 0.0020 | 0.002 | 0.95 | 0.01 | 10.9484 | **7.5601** ± 0.0602 | 3.3883 | 181.5 ms | 722,148 tok/s | 12.2% |

### 350M Transformer Sweep Results

#### CAUCHYLIFT (6 arms)

| Arm | Configuration | LR | Momentum | WD | Init Loss | Final Loss (3-Seed Mean ± Std) | Loss Drop | Latency | Throughput | MFU |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | CauchyLift LR 0.0005 | 0.0005 | 0.95 | 0.01 | 11.0253 | **8.4142** ± 0.0355 | 2.6111 | 303.5 ms | 215,921 tok/s | 10.6% |
| 2 | CauchyLift LR 0.0010 | 0.001 | 0.95 | 0.01 | 11.0253 | **7.9056** ± 0.0281 | 3.1197 | 311.4 ms | 210,441 tok/s | 10.3% |
| 3 | CauchyLift LR 0.0020 | 0.002 | 0.95 | 0.01 | 11.0253 | **7.7134** ± 0.0320 | 3.3119 | 313.2 ms | 209,263 tok/s | 10.3% |
| 4 | CauchyLift LR 0.0030 | 0.003 | 0.95 | 0.01 | 11.0253 | **7.6940** ± 0.0360 | 3.3313 | 309.3 ms | 211,851 tok/s | 10.4% |
| 5 | CauchyLift LR 0.0060 | 0.006 | 0.95 | 0.01 | 11.0253 | **7.7104** ± 0.0334 | 3.3149 | 307.0 ms | 213,473 tok/s | 10.5% |
| 6 | CauchyLift LR 0.0100 | 0.01 | 0.95 | 0.01 | 11.0253 | **7.8005** ± 0.0287 | 3.2248 | 316.2 ms | 207,264 tok/s | 10.2% |

#### ADAMW (5 arms)

| Arm | Configuration | LR | Momentum | WD | Init Loss | Final Loss (3-Seed Mean ± Std) | Loss Drop | Latency | Throughput | MFU |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | AdamW LR 0.0001 | 0.0001 | 0.95 | 0.01 | 11.0254 | **7.9281** ± 0.0363 | 3.0973 | 542.8 ms | 120,744 tok/s | 5.9% |
| 2 | AdamW LR 0.0002 | 0.0002 | 0.95 | 0.01 | 11.0253 | **7.6615** ± 0.0338 | 3.3638 | 364.3 ms | 179,914 tok/s | 8.8% |
| 3 | AdamW LR 0.0004 | 0.0004 | 0.95 | 0.01 | 11.0254 | **7.6647** ± 0.0358 | 3.3607 | 363.2 ms | 180,459 tok/s | 8.9% |
| 4 | AdamW LR 0.0008 | 0.0008 | 0.95 | 0.01 | 11.0253 | **7.6695** ± 0.0369 | 3.3558 | 357.2 ms | 183,486 tok/s | 9.0% |
| 5 | AdamW LR 0.0015 | 0.0015 | 0.95 | 0.01 | 11.0253 | **7.6772** ± 0.0357 | 3.3481 | 363.9 ms | 180,076 tok/s | 8.8% |

## Hardware and Cluster Topology
- **Hardware**: 16x Google Cloud TPU v4 chips (`v4-32` slice, 4 worker hosts)
- **Interconnect**: 2x2x4 3D Torus optical mesh
- **Software Stack**: PyTorch 2.5 + Torch-XLA 2.5 / PJRT
- **Data Pipeline**: Packed, deterministic token stream from real FineWeb-Edu partitions
