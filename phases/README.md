# Autonomous Research Phases

These nine files are copy-ready prompts for fresh Codex/agent sessions. Run them in order. A later phase may begin only when the preceding phase has committed a PASS handoff. The prompts authorize in-scope edits, local computation, experiments, non-force Git commits, and pushes to this repository; they do not authorize purchases, destructive administration, disclosure of credentials, or submission of a manuscript to a venue.

| Phase | Purpose | Expensive Hardware Work |
|---|---|---|
| 1 | Mathematical foundation and scoped novelty audit | No |
| 2 | Stochastic dynamics, fiber curvature bounds, and scaling theory | Small diagnostics only |
| 3 | PyTorch reference and native TPU/XLA fused multi-tensor kernels | Kernel benchmarks |
| 4 | High-performance decoder-only Transformer and token data system | Smoke tests |
| 5 | Small-scale multi-workload screen and baseline verification | Small training sweeps |
| 6 | Scaling pilot, 16x TPU v4-32 orchestration, and dual preregistration | Medium sweeps |
| 7 | Frozen 125M-parameter, 2.5B-token experiment on 16x TPU v4-32 | Yes (2.5B tokens) |
| 8 | Frozen 350M-parameter, 3B-token flagship experiment on 16x TPU v4-32 | Yes (3B tokens) |
| 9 | Cross-scale analysis, reproducibility audit, and publishable paper | Rechecks only |

## Shared State Machine

Every phase must read the repository, all earlier phase reports, the research contract, the future experiment protocol, the evidence ledger, the risk register, and the current paper before acting. It must inspect the current machine rather than assume package versions, available disk, or GPU availability.

Each phase writes:

- `artifacts/phaseN/report.md`: decisions, evidence, failures, and the exact gate result;
- `artifacts/phaseN/manifest.json`: commit, environment, commands, seeds, inputs, output paths, and hashes of decisive artifacts;
- `artifacts/phaseN/commands.log`: commands sufficient to reconstruct the work, with secrets redacted;
- `phases/status/phaseN.json`: PASS, REVISE, FAIL_CORE, or BLOCKED, plus reasons and invalidated downstream phases.

Generated checkpoints and dataset caches must stay outside Git. Small logs, configurations, plots, summaries, and hashes belong in Git. Preserve failed runs; never overwrite or omit them. Use UTC timestamps and deterministic run identifiers.

## Failure Routing

1. Classify every failure as implementation, infrastructure, resource, experimental-design, mathematical, novelty, or empirical.
2. Fix implementation and infrastructure failures inside the current phase and rerun the smallest decisive test.
3. A mathematical failure, a violated assumption, or evidence that a proposed mechanism is false triggers the theory-repair loop. Reopen phases 1 and 2; search current primary literature; derive first-principles alternatives; update the affected theorems, proofs, formal artifacts, paper sections, and evidence ledger; mark affected downstream evidence invalid; and rerun every invalidated gate.
4. An empirical loss is evidence, not automatically an implementation bug. Diagnose it without tuning on held-out results. If it contradicts the proposed mechanism, enter theory repair. Otherwise retain and report the negative result.
5. Resource limits may reduce a diagnostic scale, but may not be relabeled as completion of the full-scale experiments.

## Scientific and Operational Rules

- **The CauchyLift Primitive:** CauchyLift is defined as the unified coupling of:
  1. Directional velocity filtering via historical momentum ($M_t = \beta M_{t-1} + (1 - \beta) G_t$);
  2. Additive Fiber RMS Cauchy lifting ($D_{ij} = \text{RMS}(M_{i,:}) + \text{RMS}(M_{:,j})$, $Z_{ij} = M_{ij}/D_{ij}$, $U = \sqrt{\max(m, n)} Z / \|Z\|_F$);
  3. Decoupled weight decay ($W_{t+1} = W_t(1 - \eta \lambda) - \eta U$).
- It operates with linear work in parameter count ($O(N^2)$ fiber reductions, zero matrix inversions or SVD), hardware-regular operations, single-state memory overhead (4 bytes/param, 50% less memory than AdamW), and native sub-millisecond TPU / XLA HLO execution.
- Keep the primitive and all baselines faithful to their definitions. Give every optimizer the same tuning budget, data order, model, token accounting, schedule family, seed policy, and reporting standard.
- Do not optimize against the confirmatory test set or hide failed configurations. Register decisions before observing held-out outcomes.
- On the TPU server, begin with read-only inventory commands. Use the 16x TPU v4-32 slice deliberately via Torch-XLA / PJRT, run memory-heavy jobs cleanly, preserve resumable checkpoints, and never alter global drivers without explicit authority.
- Compute reductions and delicate denominator arithmetic in FP32 accumulation. Train in BF16 where supported and verified.
- Continue autonomously through safe, in-scope work. Pause only for a genuine need for credentials, money, destructive system changes, unavailable hardware, or a material change of research scope.
- Do not mark a phase PASS because code runs. PASS requires every stated gate and its evidence.
- Commit each phase with a descriptive message and push without force after validation.

## Recommended Invocation

Open a fresh session at the repository root and provide the entire relevant phase file as the request. Phases 7 and 8 are intentionally isolated so that long GPU runs can be resumed without mixing design decisions into confirmatory execution.
