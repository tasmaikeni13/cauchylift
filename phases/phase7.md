# Phase 7 prompt — frozen 125M-parameter, 1B-token experiment

Work autonomously in the CauchyLift repository and complete Phase 7. Read phases/README.md and require PASS handoffs through Phase 6. This phase executes the frozen confirmatory protocol; it does not design the optimizer.

## Objective

Train the frozen approximately 125M-parameter decoder-only Transformer for exactly 1,000,000,000 non-padding FineWeb-Edu training tokens per run on the single MI300X, for CauchyLift and every frozen baseline and seed. Produce complete, resumable, auditable evidence without post-hoc tuning.

## Required work

1. Verify the repository commit, protocol and config hashes, dataset revision and shard hashes, parameter count, partition non-overlap, token-counter tests, software environment, MI300X health, free memory, and disk margin. Refuse to run on protocol drift. Record the preflight result.
2. Use the Phase 6 serial schedule and randomized optimizer order. Run one memory-heavy job at a time. The orchestration must be idempotent, lock against duplicate runs, checkpoint atomically, resume from the exact data cursor and RNG state, and verify completed token counts before advancing.
3. Execute all frozen optimizers and at least three seeds exactly as preregistered. Every run sees the paired token stream for its seed. Do not alter learning rates, warmup, batch size, radius, epsilon behavior, validation frequency, or stop rule after inspecting outcomes.
4. Monitor loss, validation, throughput, optimizer timing, memory, update alignment, row/column concentration, effective support, stable rank, denominator extremes, boundary frequency, and loss spikes. Keep structured local logs and periodic compact summaries. Monitoring must not materially change measured throughput.
5. Retry a transient infrastructure failure from the last valid checkpoint at most twice after diagnosing it. Preserve the failed log and reason. An OOM may be repaired only by a preregistered equivalent accumulation setting; otherwise invalidate and amend the protocol before restarting every affected comparison. Never skip a failed seed.
6. If HIP and reference results diverge, a run crosses a numerical kill threshold, or a mechanism metric contradicts Phase 2, stop the affected suite and classify the failure. Implementation failures return to Phase 3. Mathematical failures invoke Phases 1 and 2, require paper and theory repair, create a new version, and invalidate the frozen suite. Do not patch the optimizer during a confirmatory run.
7. On completion, verify that each successful run consumed exactly 1B non-padding training tokens and that validation never contaminated training. Produce a checksummed run index linking configs, checkpoints, logs, summaries, environment, commit, seed, data stream, start and end times, and failure history.
8. Compute only preregistered interim summaries needed to verify completeness. Do not choose favorable subsets or write the final scientific conclusion in this phase.

## Gate

Phase 7 passes only if:

- every frozen optimizer and seed is complete or is accounted for under the preregistered crash rule;
- every successful run has the correct 125M-class parameter count and exactly 1B FineWeb-Edu training tokens;
- paired data order, validation isolation, config hashes, checkpoints, and logs validate;
- no unresolved implementation, numerical, or protocol discrepancy remains;
- all failures and retries are retained;
- raw artifacts are sufficient for an independent Phase 8 analysis.

If compute is interrupted, remain in Phase 7 and resume; a partial suite is not PASS. Keep large checkpoints and datasets outside Git, but commit their hashes, manifests, compact logs, and completion ledger. Push progress and the final validated Phase 7 artifacts without force.
