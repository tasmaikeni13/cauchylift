# Phase 6 prompt — scaling pilot and frozen 125M/1B preregistration

Work autonomously in the CauchyLift repository and complete Phase 6. Read phases/README.md and require PASS handoffs through Phase 5. This phase may use substantial but bounded MI300X time. It must freeze the final experiment before the final held-out outcomes exist.

## Objective

Test whether the small-scale result transfers toward 125M parameters, select hyperparameters with equal budgets, estimate compute and storage from measurements, and produce an immutable protocol for training an approximately 125M-parameter decoder-only Transformer on exactly 1,000,000,000 non-padding FineWeb-Edu tokens per confirmatory run.

## Required work

1. Commit a pilot protocol before execution. Use at least two increasing decoder-only model scales and token budgets between the Phase 5 models and the final run. Specify seeds, tuning partitions, optimizer grids, schedules, target losses, batch semantics, and stop rules. Do not use the final held-out partition for tuning.
2. Run equal-budget pilots for CauchyLift, AdamW, Muon, SOAP, and SinkGD when faithful implementations passed Phase 5. Keep normalized-gradient and sign controls where they remain scientifically informative. Use the same token streams and paired seeds.
3. Measure scaling of validation loss, tokens-to-target, throughput, optimizer overhead, memory, update statistics, concentration, loss spikes, and hyperparameter optima. Fit only simple predeclared scaling summaries and report uncertainty; do not extrapolate a victory from two points.
4. Resolve all final architecture choices. Generate a model configuration whose actual trainable parameter count is 125M within plus or minus 2 percent, and report total, embedding, and non-embedding counts. Freeze layer count, width, heads, sequence length, vocabulary, tokenizer, positional method, normalization, activation, weight tying, initialization, and parameter-shape policy.
5. Freeze the data protocol: exact FineWeb-Edu revision and license, deterministic 1B-token training stream, separate tuning and validation partitions, final held-out document IDs or hashes, packing rules, sequence length, and exact token-counter test. The same 1B-token sequence must be presented to every confirmatory optimizer for a given seed.
6. Freeze at least four final optimizers: CauchyLift, AdamW, Muon, and the strongest faithful non-Muon matrix or stateless baseline from the pilots. Include both SOAP and SinkGD if resources and Phase 5 validity permit. Use at least three confirmatory seeds per optimizer. No final-run hyperparameter may be selected using final-run validation outcomes.
7. Give each optimizer the same number of pilot tuning trials. Freeze learning rates, schedule and warmup, any legitimate baseline-specific documented hyperparameters, regularization policy, global batch size, gradient accumulation, precision, validation cadence, checkpoint cadence, and failure criteria. Explicitly state why the regularization policy does not disguise a composite CauchyLift update.
8. Define the primary outcome, secondary outcomes, target-loss thresholds, paired comparisons, confidence intervals, treatment of crashed runs, and the exact research-contract decision rule. Define what would constitute a mechanism contradiction and what would be an ordinary empirical loss.
9. Benchmark the frozen 125M configuration briefly to estimate tokens per second, wall time per run, total suite time, energy if measurable, checkpoint size, dataset-cache size, and worst-case disk use. Verify that the MI300X memory and local disk have safe margins. Create a resumable serial run plan with randomized optimizer order across seeds.
10. Write the immutable final configuration and protocol under experiments/protocols, record file hashes and the Git commit, and add a validator that refuses to start Phase 7 if a frozen field changes or a prerequisite artifact is missing.

## Gate

Phase 6 passes only if:

- pilot results preserve the Phase 5 advantage and show no new theory or stability failure;
- all compared methods received equal tuning budgets and all attempts are recorded;
- the exact 125M-class model, 1B-token stream, final optimizer set, at least three seeds, metrics, and analysis are frozen before final evaluation;
- a measured resource plan shows the complete serial suite can run safely on one MI300X and available disk;
- automatic validation catches protocol drift, partition overlap, wrong parameter counts, and wrong token budgets;
- the preregistration commit and artifact hashes are recorded in the Phase 6 report.

If the pilot contradicts the mechanism or kill criteria, invoke theory repair and rerun affected phases instead of weakening the final protocol. Write the standard Phase 6 artifacts. Commit and push the preregistration without force. Do not launch the 1B-token suite in this session.
