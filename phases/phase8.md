# Phase 8 prompt — statistical analysis, reproducibility audit, and publishable paper

Work autonomously in the CauchyLift repository and complete Phase 8. Read phases/README.md and require a PASS handoff from Phase 7. Analyze the frozen evidence exactly as preregistered and turn the full project into a submission-ready research artifact. Do not submit it to a venue or claim external peer review.

## Objective

Determine honestly whether CauchyLift survived its mathematical, systems, and empirical kill criteria. Produce a rigorous paper with new mathematics, scoped novelty evidence, complete optimizer comparisons, MI300X implementation evidence, limitations, and reproducible artifacts. A negative outcome must remain publishable as a negative result rather than be hidden.

## Required work

1. Validate the complete run index, hashes, token counts, partitions, configs, seeds, retries, and protocol commit before analysis. Recompute decisive summaries from raw structured logs with a tested script. Exclude nothing except by the preregistered rule, and show exclusions.
2. Perform the frozen paired statistical analysis for tokens-to-target, final validation loss, full-step and optimizer-only time, throughput, memory, loss spikes, alignment, concentration, effective support, stable rank, and denominator behavior. Report per-seed values, central estimates, uncertainty intervals, effect sizes, sensitivity, and all attempted hyperparameters. Do not treat correlated checkpoints as independent samples.
3. Separate algorithmic efficiency from systems efficiency. Compare quality per token, quality per optimizer step, and quality per wall-clock time. Explain single-MI300X results as single-hardware evidence; do not claim the original two-GPU-family performance gate was satisfied.
4. Test the mechanism predictions from Phase 2 against the logged quantities. If a claimed theorem is violated, first suspect implementation and reproduce a minimal case; if the mathematics is genuinely wrong, invoke the shared theory-repair loop, correct the faulty paper and formal section, invalidate affected conclusions, and rerun what the correction changes. If a hypothesis is simply unsupported, say so and remove causal language.
5. Refresh the novelty and related-work audit through the analysis date using primary sources and formula-level comparisons. If equivalent prior art is found, withdraw novelty claims. Use a scoped statement describing searched sources and dates; never assert omniscience over the ML community.
6. Rewrite paper/paper.md as the canonical complete manuscript. Include a precise abstract; motivation; construction; all assumptions; proofs and formal boundary; stochastic and finite-precision theory; complexity; HIP implementation; experimental protocol; all baselines; small, pilot, and 125M/1B results; ablations; mechanism tests; related work; limitations; threats to validity; broader impacts where appropriate; and conclusion.
7. Create a submission-ready typeset version and bibliography from the canonical manuscript, plus generated figures and tables whose source data and scripts are tracked. Build the PDF in a pinned environment and inspect it for broken references, clipped plots, illegible labels, incorrect equations, and claim inconsistencies.
8. Update README.md, REPRODUCIBILITY.md, CITATION.cff, the evidence ledger, proof audit, risk register, experiment protocol, and repository map. Add one clean command or documented sequence for unit tests, Lean, kernel correctness, figure regeneration, analysis regeneration, and a small end-to-end smoke run.
9. Audit the repository from a clean checkout. Run CPU tests, Lean, static checks, available ROCm tests, result regeneration, manuscript build, link checks, secret scanning, license checks, and a large-file check. Verify that ignored caches and checkpoints are not accidentally tracked.
10. Create a claim-to-evidence table labeling every major sentence as proved, formally checked, numerically checked, measured on MI300X, hypothesized, or not established. Remove marketing language, causal claims unsupported by interventions, and any claim that depends on a hidden optimizer combination.

## Final decision

Use the original frozen kill criteria. If CauchyLift clears them, state the exact scope: model sizes, 1B-token budget, FineWeb-Edu revision, baselines, tuning budget, seed count, and single MI300X. If it does not, clearly reject the performance hypothesis while retaining correct mathematical and systems contributions. Do not invent a positive conclusion or tune on confirmatory outcomes.

## Gate

Phase 8 passes only if:

- every table and figure is reproducible from tracked scripts and checksummed raw evidence;
- the manuscript agrees with the preregistration and includes all failures and negative results;
- mathematical statements, Lean boundaries, implementation behavior, and empirical claims are mutually consistent;
- novelty language survives the refreshed scoped audit;
- the clean-checkout test and manuscript build pass;
- the paper is submission-ready, though no external submission has occurred.

Write the standard Phase 8 artifacts and a concise final research verdict. Commit and push validated source, manuscript, plots, and small evidence without force. Tag a version only if the repository policy permits it and all gates pass; never force-push or delete the history of failed hypotheses.
