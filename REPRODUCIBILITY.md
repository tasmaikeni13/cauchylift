# Reproducibility record

Artifact freeze dates: 2026-08-27 UTC for the original theory suite; 2026-08-28 UTC for Phases 1 and 2.

## Environment used for checked-in results

- Python 3.12.3
- no third-party Python packages
- Lean 4.19.0
- mathlib tag `v4.19.0`, resolved commit recorded in `formal/lake-manifest.json`
- random seed `20260827` unless an artifact says otherwise

## Regenerate analysis outputs

From the repository root:

```bash
python3 analysis/run_property_checks.py \
  --samples 5000 \
  --output analysis/results/property_checks.json

python3 analysis/run_quadratic_suite.py \
  --trials 16 \
  --exact-steps 600 \
  --scheduled-steps 400 \
  --output analysis/results/quadratic_suite.json

python3 analysis/run_rank_probe.py \
  --samples 200 \
  --output analysis/results/rank_probe.json

python3 analysis/run_rejection_checks.py \
  --trials 200 \
  --output analysis/results/rejection_checks.json

python3 analysis/run_adversarial_audit.py \
  --output analysis/results/adversarial_audit.json

python3 analysis/run_boundary_suite.py \
  --seed 20260828 \
  --samples 10000 \
  --output analysis/results/boundary_suite.json

python3 analysis/run_stochastic_suite.py \
  --output analysis/results/stochastic_suite.json

python3 analysis/run_mechanism_suite.py \
  --output analysis/results/mechanism_suite.json

python3 analysis/run_width_suite.py \
  --spec spec/optimizer_v0.2.json \
  --output analysis/results/width_suite.json

python3 analysis/run_finite_precision_suite.py \
  --output analysis/results/finite_precision_suite.json
```

The scripts use deterministic pseudo-random generators and sorted JSON keys. Re-running them in the recorded Python version should reproduce the checked-in files byte for byte.

## Check the formal artifact

```bash
cd formal
lake update
lake build
```

The successful build target is `CauchyLift`. See `formal/README.md` and `research/proof_audit.md` before interpreting the scope.

## Fast local audit

```bash
python3 -m compileall -q analysis
python3 analysis/run_property_checks.py --samples 5000 >/dev/null
(python3 analysis/run_boundary_suite.py --samples 1000 >/dev/null)
(cd formal && lake build)
```

No command in this record trains a model or downloads a dataset.
