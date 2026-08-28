# Predeclared future empirical protocol

This document is a protocol only. No training code is included in this repository.

## Primary question

At equal tuning effort, does the unmodified CauchyLift primitive reduce tokens or examples to a fixed validation target while keeping optimizer-step cost within 15% of AdamW?

## Baselines

- AdamW
- Muon with its standard Newton–Schulz configuration
- SOAP at its documented preconditioning frequency
- SinkGD with five normalization rounds
- normalized gradient descent and sign descent as mechanism controls

No baseline may receive less tuning budget, fewer seeds, or a weaker learning-rate schedule family.

## Workloads

Predeclare at least four workloads before implementation:

1. a small decoder-only language model where full sweeps are affordable;
2. a medium decoder-only model with fixed token budget;
3. a vision transformer;
4. a convolutional or state-space architecture that tests non-square tensors.

At least one workload must be held out from all method design and initial hyperparameter choices.

## Metrics

- tokens/examples to each predeclared training-loss and validation-loss target;
- end-of-budget validation metric;
- optimizer-only wall time and full-step wall time;
- peak persistent and transient memory;
- update cosine with the raw gradient;
- row/column concentration, effective support, and update stable rank;
- loss-spike count under a predeclared threshold;
- sensitivity curves over learning rate and radius scaling.
- minibatch gradient SNR estimates \(\hat\sigma/\|\hat\mu\|_F\), conditional-alignment margin, and observed expected-alignment failures;
- local two-mode log-ratio slope and alternating update concentration where a two-mode approximation is measurable;
- exact-boundary and FP64 rare-path counts.

## Tuning

- identical log-spaced learning-rate budget per method;
- identical schedule families and warmup choices;
- at least three seeds for screening and five for confirmatory runs;
- selection on training workloads, final reporting on the held-out workload;
- report all attempted configurations, including failures.

## Ablations that do not redefine the method

- denominator computed in FP32 versus FP64 diagnostic arithmetic;
- fused versus unfused implementation;
- the frozen radius \(\sqrt{\min(m,n)}\) may be compared diagnostically with alternatives, but the confirmatory CauchyLift result uses the frozen value from `spec/optimizer_v0.2.json`;
- exact projective-boundary branch frequency.

Momentum, moments, rotation, clipping, blending, or periodic polar steps are **not** ablations; they create different composite optimizers and cannot rescue the core hypothesis under this contract.

## Frozen parameter coverage

The initial decoder is bias-free. Every trainable tensor—including embeddings or a tied head, attention/MLP matrices, normalization gains, and scalars—uses the matrixization and zero/boundary semantics in `spec/optimizer_v0.2.json`. Sparse layouts may use an exactly equivalent support-aware implementation, but there is no Adam/SGD fallback. Any need for a different optimizer on a parameter class is a contract-level negative result.

## Decision rule

Promote CauchyLift from theory hypothesis only if it clears the kill criteria in `research_contract.md`. Otherwise publish the negative result and retain the mathematical artifact without performance marketing.
