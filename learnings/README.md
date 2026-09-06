# OpenBoost Learnings

This directory is the repository's durable engineering memory. It records why a
change was made, what evidence supports it, which attempts failed, and what
remains unknown. It is deliberately separate from release notes and generated
benchmark output.

## When to Write an Entry

Create or update an entry when work includes any of the following:

- a non-obvious correctness fix;
- a benchmark, experiment, or falsified hypothesis;
- an architecture or product-scope decision;
- a dependency, platform, CI, packaging, or release incident;
- a user correction that should change future agent behavior.

Use `YYYY-MM-DD-short-topic.md`. Prefer one entry per coherent investigation;
append follow-up evidence rather than creating many tiny diary files.

## Required Content

Start from `TEMPLATE.md` and include:

- context and the question being answered;
- decision or result;
- files/behavior changed;
- verification and artifact locations;
- failed attempts and why they failed;
- remaining risks and next action.

Keep entries factual and concise. Do not include credentials, secrets, private
URLs, or copied raw logs. Link the relevant commit after it exists.

## Current Entries

- [English repository prose](2026-09-06-english-repository.md) — repository-wide
  translation with unchanged scope, gates, and historical evidence.

- [Adult official test and stratified splits](2026-09-06-v1-adult-data.md) — A2 raw
  data freeze, with missing categories retained and fnlwgt excluded.

- [Housing five splits](2026-09-06-v1-housing-data.md) — A1/A11 share verified
  legacy inputs plus seeds 3–4; licensing and quality remain unresolved.

- [A5 Bike data freeze](2026-09-06-v1-bike-data.md) — verified archive, calendar-only
  inputs and five complete-date rolling windows; no model quality result yet.

- [v1 artifact integrity](2026-09-06-v1-artifact-integrity.md) — missing cells,
  stale cache identities and false pass claims fail; quality gates remain unevaluated.

- [Reference integration and F0.2 exit](2026-09-06-v1-reference-integration-exit.md) —
  finite model/state compositions, 288 tests; next F0.3, production gates pending.

- [Retire legacy production](2026-09-05-retire-legacy-production.md) — user-directed
  clean v1 package reset; historical code at `50acfc6`, current training API pending.

- [v1 sprint execution](2026-09-05-v1-sprint-execution.md) — sprint plans/results
  and reflection now live in `v1-sprints/`; starts B01/F0.2 scalar/tree references.

- [Foundation construction design](2026-09-05-foundation-construction-design.md) —
  concrete module/data/operation/state/device contracts and B01–B14 build slices;
  distinguishes the engineering design from task cards and independent oracles.

- [F0.1 foundation task cards](2026-09-05-foundation-f0-task-cards.md) —
  all A1–A13 tasks specified, independent failure cases and fair baseline paths;
  F0.2 is next, runtime and benchmark gates remain unverified.

- [OpenBoost v1 scope, releases and evaluation](2026-09-05-openboost-v1-plan.md) —
  real v1 planning baseline; all A1–A13 use cases individually required, no
  privileged insurance/AFT focus; releases/plans and quantitative acceptance.

- [Foundation product and clean redesign](2026-09-05-agent-foundation-reset.md) —
  latest user direction, supersedes the earlier risk-first investment ordering;
  active F0–F5 plan, no backward compatibility requirement.

- [Independent GPU extension wheels](2026-09-05-foundation-p6-gpu-wheels.md)

- [Strict CUDA extension trainer](2026-09-05-foundation-p5-trainer.md)

- [Independent CPU extension wheels](2026-09-05-foundation-p6-cpu-wheels.md)

- [P4.4 level-wise builder and T4 evidence](2026-09-05-foundation-p4-builder.md)

- [2026-09-05-foundation-p4-leaves.md](2026-09-05-foundation-p4-leaves.md) — real-row CPU/CUDA leaf reduction, explicit leaf rule, bounded next-gradient checks and real T4 evidence.

- [2026-09-05-foundation-p4-splits-goal-review.md](2026-09-05-foundation-p4-splits-goal-review.md) — goal/value checkpoint, earlier independent CPU package validation, numeric split/routing and real T4 evidence.

- [2026-09-05-foundation-p4-histograms.md](2026-09-05-foundation-p4-histograms.md) — fixed-slot CPU/CUDA histogram contract, independent sample oracle and real T4 validation.

- [2026-09-05-foundation-p3-cpu-contract.md](2026-09-05-foundation-p3-cpu-contract.md) — experimental CPU objective/builder/schedule, plugin-free wheel inference, coefficient persistence and installation limits.
- [2026-09-05-foundation-p2-boundaries-baseline.md](2026-09-05-foundation-p2-boundaries-baseline.md) — real T4 boundary verification and frozen California Housing baseline; P2 complete.
- [2026-09-05-foundation-p2-correctness.md](2026-09-05-foundation-p2-correctness.md) — real T4 weighted regression before/after evidence, scoped RNG and explicit execution boundaries.
- [2026-09-05-foundation-p1-modal.md](2026-09-05-foundation-p1-modal.md) — isolated Modal wheel smoke, provenance checks, and explicit failure propagation.
- [2026-09-05-foundation-p0-integration.md](2026-09-05-foundation-p0-integration.md) — integrate unified trainer, preserve local fixes, and cover generic loading of the new models.
- [2026-09-05-gpu-python-foundation-design.md](2026-09-05-gpu-python-foundation-design.md) — scoped GPU Python foundation design, branch integration, experimental contracts, and medium execution checklist.
- [2026-09-05-impact-adoption-value-strategy.md](2026-09-05-impact-adoption-value-strategy.md) — impact/adoption/value research, current online versus local evidence, and proposed validation gates.
- `2026-08-15-repository-audit.md` — product focus, correctness risks, and
  evidence gaps found in the deep audit.
- `2026-08-15-scoringbench-integration.md` — third-party benchmark integration,
  validation, and Intel macOS runtime limitation.

- [2026-09-05: v1 data and classification references](2026-09-05-v1-data-classification-reference.md) — independent column transforms, geometry and two-round checks.

- [2026-09-05: ranking, quantile and vector references](2026-09-05-v1-ranking-quantile-vector-reference.md) — separate pair geometry, split statistics and leaf solvers.

- [2026-09-05: positive targets and AFT](2026-09-05-v1-positive-aft-reference.md) — exposure, paid-loss definitions and stable censored geometry.

- [2026-09-05: Normal and Formula references](2026-09-05-v1-normal-formula-reference.md) — direction fitting and immutable candidate updates.

- [2026-09-05: identity and isolated runs](2026-09-05-v1-identity-runs-reference.md) — prepared row binding, stable RNG and comparable selection.

- [2026-09-05: exact author mutation references](2026-09-05-v1-author-mutation-reference.md) — expectile, penalized quantile and ordered acceptance.

- [2026-09-05: mixed/vector growth references](2026-09-05-v1-mixed-vector-growth-reference.md) — fitted transforms, multilevel routing and full vector leaves.

- [CPU/CUDA comparator preflight](2026-09-06-v1-comparator-capabilities.md): installed support, native failures and isolated T4 evidence.
- [Validation-only baseline worker](2026-09-06-v1-baseline-worker.md): explicit inputs, saved offsets and the remaining search integration boundary.
- [Validation selection and test release](2026-09-06-v1-validation-selection.md): independent scores, complete trials, pinned receipts and access-sequence limitations.
- [Native baseline early stopping](2026-09-06-v1-early-stopping.md): weighted validation, selected iteration persistence and new-process counterexamples.
- [Query-aware ranking worker](2026-09-06-v1-ranking-worker.md): explicit group weights, partition checks and named metric selection.
- [Parametric and paid-loss workers](2026-09-06-v1-parametric-workers.md): explicit target units, positive-payment reconstruction and global-formula controls.
- [Auxiliary quality metrics](2026-09-06-v1-auxiliary-metrics.md): calibration/class diagnostics, supported IPCW scoring and descriptive paired intervals.

- [A6 target scaling](2026-09-06-v1-target-scaling.md): train-only normalization and saved original-unit predictions.

- [Housing hosted license](2026-09-06-v1-source-license-review.md): matched archive and dated attribution evidence.
