# Distributed Pretraining Sweep Summary: 125M Transformer on 2.5B FineWeb-Edu Tokens

**Hardware:** 16-Chip Google Cloud TPU v4 Pod Slice (`v4-32`, 32 TensorCores, 4 Hosts, 2x2x4 3D Torus)

### Configuration Rankings by Mean Validation Loss

| Rank | Configuration | Completed Seeds | Mean Val Loss | Cross-Seed Std | Mean Train Loss | Throughput (tok/s) | MFU (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ADAMW (LR = 0.0020)** | 3/3 | **3.1303** | ±0.0031 | 3.1155 | 1,021,742 | 17.2% |
| 2 | **CAUCHYLIFT (LR = 0.0025)** | 3/3 | **3.2228** | ±0.0008 | 3.2004 | 1,016,928 | 17.1% |

### All Individual Runs

| Run Name | Optimizer | LR | Seed | Tokens Evaluated | Best Val Loss | Final Train Loss | Tok/s | MFU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `125m_adamw_lr0.0020_seed42` | adamw | 0.002 | 42 | 2,500,067,328 | 3.1269 | 3.3812 | 1,008,622 | 17.0% |
| `125m_adamw_lr0.0020_seed43` | adamw | 0.002 | 43 | 2,500,067,328 | 3.1330 | 3.0300 | 1,052,891 | 17.7% |
| `125m_adamw_lr0.0020_seed44` | adamw | 0.002 | 44 | 2,500,067,328 | 3.1309 | 2.9351 | 1,003,711 | 16.9% |
| `125m_cauchylift_lr0.0025_seed42` | cauchylift | 0.0025 | 42 | 2,500,067,328 | 3.2237 | 3.4806 | 1,015,653 | 17.1% |
| `125m_cauchylift_lr0.0025_seed43` | cauchylift | 0.0025 | 43 | 2,500,067,328 | 3.2221 | 3.1049 | 1,018,586 | 17.2% |
| `125m_cauchylift_lr0.0025_seed44` | cauchylift | 0.0025 | 44 | 2,500,067,328 | 3.2228 | 3.0156 | 1,016,546 | 17.1% |
