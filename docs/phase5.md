# Phase 5: Small-Scale Falsification and Baseline Screen

Phase 5 evaluates the frozen CauchyLift v0.2 optimizer primitive against AdamW, Muon, SOAP, SinkGD, NormalizedGD, and SignDescent on four low-cost workloads under an immutable, pre-committed equal-budget protocol.

## Protocol & Workloads

Protocol definition: [`experiments/protocols/phase5_protocol.json`](../experiments/protocols/phase5_protocol.json)
Protocol SHA256: `d2228adaee66bdeadffc841aedf989a5edf6084c82d44799990bb5ebf41622ef`

Workloads evaluated:
1. **Small Decoder LM** (`small_decoder_lm`): 4-layer Transformer, hidden dim 128, 4 heads, seq len 256 on WikiText-103.
2. **Medium Decoder LM** (`medium_decoder_lm`): 6-layer Transformer, hidden dim 256, 8 heads, seq len 256 on WikiText-103.
3. **Small Vision Transformer** (`small_vit`): 4-layer ViT, hidden dim 128, 4 heads, patch size 4 on CIFAR-10.
4. **Held-Out ConvSSM** (`conv_ssm_heldout`): 4-layer Non-Square Conv/State-Space model, held out from hyperparameter selection.

## Reproduction Commands

```bash
# 1. Run all tests (including Phase 5 models)
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build .venv/bin/pytest

# 2. Run Phase 5 screen (170 runs across tuning, confirmation, held-out, ablations)
PYTHONPATH=. CAUCHYLIFT_BUILD_DIR=/tmp/cauchylift-hip-build .venv/bin/python scripts/run_phase5_screen.py --device cuda

# 3. Generate analysis plots
PYTHONPATH=. .venv/bin/python scripts/plot_phase5_results.py
```

## Key Findings & Gate Result

- **Gate Result:** `FAIL_CORE`
- **Confirmation Validation Loss:**
  - Small Decoder LM: AdamW 7.1080 vs CauchyLift 7.9041
  - Medium Decoder LM: AdamW 7.1427 vs CauchyLift 7.9336
  - Small ViT: AdamW 1.9226 vs CauchyLift 1.9428
  - Held-Out ConvSSM: AdamW 7.5853 vs CauchyLift 8.1187
- **Mathematical Cause:** For $m, n \gg 1$, row and column energy dispersion causes the cotransverse complement denominator $D_{ij} = (\|G\|_F^2 - \|G_{i,:}\|^2) + (\|G\|_F^2 - \|G_{:,j}\|^2)$ to become virtually uniform across matrix coordinates ($D_{ij} \approx 2\|G\|_F^2$). Consequently, CauchyLift collapses analytically and empirically to Normalized Gradient Descent without momentum.
- **Contract Adherence:** Per the Research Contract kill criteria, the core hypothesis is falsified without unprincipled rescues (no momentum, Adam moments, or clipping added). Downstream phases (6, 7, 8) are invalidated until theory repair.
