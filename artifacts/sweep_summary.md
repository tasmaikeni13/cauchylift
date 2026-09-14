# Distributed Pretraining Sweep Summary: 125M Transformer on 3B FineWeb-Edu Tokens

**Hardware:** 16-Chip Google Cloud TPU v4 Pod Slice (`v4-32`, 32 TensorCores, 4 Hosts, 2x2x4 3D Torus)

### Configuration Rankings by Mean Validation Loss

| Rank | Configuration | Completed Seeds | Mean Val Loss | Cross-Seed Std | Mean Train Loss | Throughput (tok/s) | MFU (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **CAUCHYLIFT (LR = 0.0025)** | 1/3 | **3.1885** | ±0.0000 | 3.0448 | 1,054,695 | 17.8% |

### All Individual Runs

| Run Name | Optimizer | LR | Seed | Tokens Evaluated | Best Val Loss | Final Train Loss | Tok/s | MFU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `125m_cauchylift_lr0.0025_seed42` | cauchylift | 0.0025 | 42 | 3,000,238,080 | 3.1885 | 3.0448 | 1,054,695 | 17.8% |
