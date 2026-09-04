# Autonomous research phases

These nine files are copy-ready prompts for fresh Codex/agent sessions. Run them in order. A later phase may begin only when the preceding phase has committed a PASS handoff. The prompts authorize in-scope edits, local computation, experiments, non-force Git commits, and pushes to this repository; they do not authorize purchases, destructive administration, disclosure of credentials, or submission of a manuscript to a venue.

| Phase | Purpose | Expensive GPU work |
|---|---|---|
| 1 | Adversarial novelty and mathematical audit | No |
| 2 | Stochastic, boundary, scale, and full-model theory | Small diagnostics only |
| 3 | PyTorch reference and native ROCm/HIP optimizer kernels | Kernel benchmarks |
| 4 | Optimized decoder-only Transformer and data system | Smoke tests |
| 5 | Small-scale falsification and baseline screen | Small training sweeps |
| 6 | Scaling pilot, 8x MI300X orchestration, and dual preregistration | Medium sweeps |
| 7 | Frozen 125M-parameter, 1B-token experiment on 8x MI300X | Yes (1B tokens) |
| 8 | Frozen 350M-parameter, 3B-token flagship experiment on 8x MI300X | Yes (3B tokens) |
| 9 | Cross-scale analysis, reproducibility audit, and publishable paper | Rechecks only |

## Shared state machine

Every phase must read the repository, all earlier phase reports, the research contract, the future experiment protocol, the evidence ledger, the risk register, and the current paper before acting. It must inspect the current machine rather than assume package versions, available disk, or GPU availability.

Each phase writes:

- artifacts/phaseN/report.md: decisions, evidence, failures, and the exact gate result;
- artifacts/phaseN/manifest.json: commit, environment, commands, seeds, inputs, output paths, and hashes of decisive artifacts;
- artifacts/phaseN/commands.log: commands sufficient to reconstruct the work, with secrets redacted;
- phases/status/phaseN.json: PASS, REVISE, FAIL_CORE, or BLOCKED, plus reasons and invalidated downstream phases.

Generated checkpoints and dataset caches must stay outside Git. Small logs, configurations, plots, summaries, and hashes belong in Git. Preserve failed runs; never overwrite or omit them. Use UTC timestamps and deterministic run identifiers.

## Failure routing

1. Classify every failure as implementation, infrastructure, resource, experimental-design, mathematical, novelty, or empirical.
2. Fix implementation and infrastructure failures inside the current phase and rerun the smallest decisive test.
3. A mathematical failure, a violated assumption, a formula collision, or evidence that the proposed mechanism is false triggers the theory-repair loop. Reopen phases 1 and 2; search current primary literature; derive multiple first-principles alternatives; reject anything compositional; update the faulty theorem, proof, formal artifact, paper section, and evidence ledger; mark affected downstream evidence invalid; and rerun every invalidated gate.
4. Never rescue the method by adding momentum, moments, clipping, rotation, whitening, a polar step, an Adam/SGD fallback, a baseline mixture, or another familiar optimizer module. A materially changed primitive is a new version and needs a new novelty audit and preregistration.
5. An empirical loss is evidence, not automatically a mathematical bug. Diagnose it without tuning on held-out results. If it contradicts the proposed mechanism, enter theory repair. Otherwise retain and report the negative result.
6. Resource limits may reduce a diagnostic scale, but may not be relabeled as completion of the full-scale experiments.
7. Absolute historical novelty cannot be proved by search. The acceptable claim is a dated, reproducible, scoped collision audit. If equivalent prior art is found, stop novelty claims and replace or retire the primitive.

## Scientific and operational rules

- The fixed objective is a genuinely non-compositional optimizer primitive with new mathematical structure, linear work in parameter count, GPU-regular operations, no persistent optimizer state, and credible competitive behavior on language-model pretraining.
- Keep the primitive and all baselines faithful to their definitions. Give every optimizer the same tuning budget, data order, model, token accounting, schedule family, seed policy, and reporting standard.
- Do not optimize against the confirmatory test set or hide failed configurations. Register decisions before observing held-out outcomes.
- Use every available research, theorem-proving, coding, and documentation capability internally. Record queries, dates, versions, URLs, and formula-level comparisons; scientific artifacts discuss methods and evidence, never assistant tooling.
- On the AMD server, begin with read-only inventory commands. Use the 8x MI300X cluster deliberately via PyTorch DDP / `torchrun`, run one memory-heavy job at a time, preserve resumable checkpoints, and never alter global drivers or ROCm without explicit authority.
- Compute reductions and delicate denominator arithmetic in FP32 or higher unless measured evidence justifies another choice. Train in BF16 where supported and verified.
- Continue autonomously through safe, in-scope work. Pause only for a genuine need for credentials, money, destructive system changes, unavailable hardware, or a material change of research scope.
- Do not mark a phase PASS because code runs. PASS requires every stated gate and its evidence.
- Commit each phase with a descriptive message and push without force after validation. Never rewrite or delete prior evidence.

## Recommended invocation

Open a fresh session at the repository root and provide the entire relevant phase file as the request. Phases 7 and 8 are intentionally isolated so that long GPU runs can be resumed without mixing design decisions into confirmatory execution.
