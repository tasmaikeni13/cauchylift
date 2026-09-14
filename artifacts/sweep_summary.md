# Distributed Pretraining Sweep Summary: 125M Transformer on 2.5B FineWeb-Edu Tokens

**Hardware:** 16-Chip Google Cloud TPU v4 Pod Slice (`v4-32`, 32 TensorCores, 4 Hosts, 2x2x4 3D Torus)

### Configuration Rankings by Mean Validation Loss

| Rank | Configuration | Completed Seeds | Mean Val Loss | Cross-Seed Std | Mean Train Loss | Throughput (tok/s) | MFU (%) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **ADAMW (LR = 0.0020)** | 3/3 | **3.1303** | ±0.0031 | 3.1155 | 1,031,711 | 17.4% |
| 2 | **CAUCHYLIFT (LR = 0.0010)** | 3/3 | **3.4093** | ±0.0022 | 3.3812 | 1,014,879 | 17.1% |

### All Individual Runs

| Run Name | Optimizer | LR | Seed | Tokens Evaluated | Best Val Loss | Final Train Loss | Tok/s | MFU (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `125m_cauchylift_lr0.0010_seed42` | cauchylift | 0.001 | 42 | 2,500,067,328 | 3.4083 | 3.6550 | 1,017,042 | 17.1% |
| `125m_cauchylift_lr0.0010_seed43` | cauchylift | 0.001 | 43 | 2,500,067,328 | 3.4118 | 3.2989 | 1,012,719 | 17.1% |
| `125m_cauchylift_lr0.0010_seed44` | cauchylift | 0.001 | 44 | 2,500,067,328 | 3.4078 | 3.1896 | 1,014,877 | 17.1% |
| `125m_adamw_lr0.0020_seed42` | adamw | 0.002 | 42 | 2,500,067,328 | 3.1269 | 3.3812 | 1,009,309 | 17.0% |
| `125m_adamw_lr0.0020_seed43` | adamw | 0.002 | 43 | 2,500,067,328 | 3.1330 | 3.0300 | 1,038,597 | 17.5% |
| `125m_adamw_lr0.0020_seed44` | adamw | 0.002 | 44 | 2,500,067,328 | 3.1309 | 2.9351 | 1,047,227 | 17.6% |
