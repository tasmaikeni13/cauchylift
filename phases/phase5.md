# Phase 5 prompt — small-scale falsification and optimizer screen

Work autonomously in the CauchyLift repository and complete Phase 5. Read phases/README.md and require PASS handoffs through Phase 4. This phase is designed to kill weak hypotheses cheaply, not to produce a flattering curve.

## Objective

Run predeclared, equal-budget small experiments that expose instability, bad scaling, fragile hyperparameters, and mechanism failures before any 125M run. Compare CauchyLift with AdamW and several structurally distinct optimizers under the same data, models, schedules, and reporting.

## Required work

1. Before training, write and commit a small-scale protocol with immutable run IDs, model configurations, datasets or data slices, token/example budgets, validation cadence, target losses, seeds, learning-rate and allowed method-specific grids, schedule families, global batch sizes, precision, stopping rules, and statistical summaries. Hash the protocol. Do not edit it after results arrive; append amendments that invalidate affected runs.
2. Use at least four low-cost workloads matching the existing research contract: a small decoder-only language model, a medium but still affordable decoder-only model, a small vision transformer, and a non-square convolutional or state-space workload. Hold one workload out from method design and initial hyperparameter choices.
3. Compare the unmodified CauchyLift primitive against AdamW, Muon, SOAP, SinkGD, normalized gradient descent, and sign descent when each faithful implementation is viable on ROCm. Give every method the same number of tuning trials, seeds, schedule choices, data order, and early-stopping opportunities. Record unsupported or failed baselines rather than silently dropping them.
4. Use at least three screening seeds for configurations that reach the confirmation stage. Select hyperparameters only on declared tuning partitions. Do not inspect the held-out workload until selection is frozen.
5. Measure tokens or examples to all predeclared targets, end-of-budget validation, optimizer and full-step time, throughput, peak persistent and transient memory, loss spikes, run failures, gradient/update alignment, concentration, effective support, stable rank, denominator statistics, and sensitivity to learning rate and the Phase 2 radius scaling.
6. Run only allowed ablations: FP32 versus higher-precision diagnostic denominator arithmetic, fused versus reference implementation, predeclared radius choices motivated by theory, and projective-boundary diagnostics. Momentum, moments, clipping, rotation, whitening, weight mixing, or polar steps create a different optimizer and are forbidden as rescues.
7. Plot every attempted configuration, not just winners. Produce paired comparisons, uncertainty intervals across seeds, hyperparameter sensitivity surfaces, and a failure table. Separate optimizer quality from kernel speed.
8. Diagnose failures. Reproduce implementation anomalies with a tiny deterministic case. If observed stochastic misalignment, concentration, or width behavior contradicts Phase 2, treat that as a theory failure, reopen Phases 1 and 2, edit the affected paper section and proofs, version the primitive if necessary, and rerun invalidated phases. Ordinary negative performance must not be mislabeled as a coding bug.

## Gate

Phase 5 passes only if:

- all predeclared runs and failures are accounted for and reproducible;
- no unresolved NaN, loss-spike, boundary, concentration, or HIP/reference discrepancy remains;
- CauchyLift beats tuned AdamW in tokens-to-target on at least three of the four predeclared workloads, as required by the frozen contract;
- the advantage does not depend on unequal tuning or a forbidden auxiliary mechanism;
- optimizer-step performance remains inside the Phase 3 bound and task-level gains survive the fused implementation;
- the held-out result supports rather than reverses the selected mechanism prediction.

If this gate fails, mark FAIL_CORE or REVISE and follow the shared routing rules; do not proceed merely because a 125M run was requested. Write the standard Phase 5 artifacts with raw logs, protocol hash, plots, and a decision table. Commit and push validated work without force. Do not run the scaling pilot in this session.
