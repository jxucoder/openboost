# OpenBoost Agent Guide

This is the canonical repository guidance for coding agents and automated
contributors. Tool-specific instruction files should point here instead of
duplicating policy.

## Mission

OpenBoost is a **programmable boosting foundation for researchers and agents**.
The product hypothesis is that readable, composable algorithm components and
verified CPU/CUDA execution reduce the cost of making a correct algorithm change.
Standard GBDT, NaturalBoost/NGBoost-style methods, FormulaBoost, and train-many
are use cases that determine and test the foundation's abstraction boundaries.
Real applications shape the design alongside algorithm families: classification,
regression, ranking, quantiles, multi-output, counts/positive targets, survival,
distributional and structured models, and model selection. All listed use-case
families are required v1 scope, with individual implementation and evaluation
evidence. Do not privilege insurance/AFT or substitute a few representative
successes for complete coverage. Concrete datasets may be selected; required
use cases may not be dropped. See the A1–A13
[application contracts](planning/foundation-application-contracts.md).

The user designates this planning round as the real **OpenBoost v1**. Its active
design and execution order is
[`planning/agent-boosting-foundation-plan.md`](planning/agent-boosting-foundation-plan.md).
The concrete construction design is
[`planning/foundation-construction-design.md`](planning/foundation-construction-design.md):
module dependencies, data/state records, composable operations, CPU/CUDA execution,
and B01–B14 build slices. Task specifications and independent oracles do not
constitute the foundation implementation; public components and recipes start in F1.
Execution plans, results and periodic reflections live in
[`v1-sprints/`](v1-sprints/README.md). Read the current sprint before implementing.
Reflect at sprint closure, every three implementation commits, phase transitions,
or architectural/correctness counterexamples; record evidence and deviations there.
Keep cross-sprint learnings in `learnings/` with links to the detailed sprint record.
Required scope is R1–R9/C1–C7/A1–A13; acceptance and quantitative evaluation are in
[`planning/openboost-v1-evaluation.md`](planning/openboost-v1-evaluation.md).
Use the [current release/plan review](planning/boosting-release-review-2026-09-05.md)
for XGBoost, CatBoost and LightGBM baselines; distinguish shipped features from
experimental capabilities, maintainer plans and feature requests.
The user explicitly permits a clean redesign: existing APIs, trainers, internal
representations, and persistence formats need not remain backward compatible.
Preserve mathematical correctness cases and reproducible evidence, not obsolete
interfaces. This is design permission, not a claim that the new architecture exists.

Do not position the repository as a drop-in replacement for XGBoost, LightGBM,
or CatBoost. A standard recipe does not establish full feature, quality, or speed
parity. Existing experimental capabilities remain experimental until committed,
reproducible artifacts verify their declared scope.

## Start Here

Before a non-trivial change:

1. Read this file and the relevant entries in `learnings/`.
2. Check `git status --short --branch`; preserve unrelated user changes.
3. Read the implementation, its tests, and the public documentation together.
4. Write a short plan for work spanning three or more meaningful steps.
5. Identify the smallest test that can fail before editing.

Do not trust phase comments, docstrings, README claims, or green CI as proof by
themselves. Verify the actual call path and the tests that exercise it.

## Current Priority Order

Current execution is governed by the user-approved
[Sprint 085 foundation-focus amendment](v1-sprints/085-foundation-focus-amendment.md).
The [086 next execution plan](v1-sprints/086-next-execution-plan.md) decomposes
that work into fields/histograms, candidate operations and resident training;
it freezes the next aggregation fixtures and the run-2 retrospective boundary.
Prioritize [069 authoring preparation](v1-sprints/069-authoring-pilot.md) and bounded
[078 scalar CUDA feasibility](v1-sprints/078-cuda-scalar-path.md). Prepare existing
D1 expectile control and D2 cohort-feasibility deep change; D2 must later exercise
the same programmable device boundary. No new agents are authorized by this card.
Actual independent accounting/isolation must precede author attempts; never count
designer work as independent author evidence or inspect sealed H1/H2 contents.

The user explicitly approved bounded B12/F3.1 feasibility before formal F2/E5
completion. This supersedes the earlier unadopted-overlap proposal. Audit relevant
065/068 ownership on the actual device path. Experimental CUDA storage, named
fields and routed histograms pass 33 real T4 tests at `ad2f4e6`; see the
[078-A evidence](benchmarks/v1/evidence/cuda-aggregation-078/README.md).
The separately approved [087 split slice](v1-sprints/087-cuda-split-operations.md)
passes 88 real T4 tests at `9ce790e`: 55 candidate/feasibility/route/leaf checks
plus all 33 previous cases. D2's independent cohort minima change the winning
split through public device operations; named/reordered information also passes.
See the [078-B evidence](benchmarks/v1/evidence/cuda-splits-078/README.md).
[088 resident scalar training](v1-sprints/088-resident-scalar-training.md) now has
real T4 evidence at clean `c415755`: 188 of 202 checks pass and 14 fail. All prior
88 regressions pass; dedicated ownership, rollback and saved CPU inference checks
pass. Weighted/missing split selection and two recipe prediction checks fail, so
the scalar training gate is not accepted. See the
[run-4 evidence and retrospective](benchmarks/v1/evidence/cuda-resident-078/README.md).
The explicitly approved 47-file private upload and single bounded invocation are
complete. All four hardware allowances are consumed; no retries are authorized.
Stop at the planned retrospective before broader construction. The next bounded
correction is score symmetry under swapped child summaries, with direct device
score/code-generation diagnostics and all original cases retained. CPU arithmetic
shows a compatible rounding mechanism; the actual GPU instruction sequence was
not captured. No production fix is included in this result. Further hardware
checks require a new concrete source freeze and upload/compute allowance.
Retain the 065/068 contracts. Passing subsets do not pass full boosting conformance. Formal
R1/R4/R5/R6/R8 device scope, R9, P7 and E4 remain required; two-round scalar parity
does not pass them. See the main plan amendment for the exception to F2→F3 entry.

Pause the next OpenBoost A6 configuration-05 CPU probe, wider 400-job CPU search
expansion and speculative CPU optimization. Sprint 070 remains open and supports
correctness/isolation needs of the active work; its complete ledger, real selection,
source gaps and author-accounting obligations are not removed. CPU is the semantic
reference and usable development path, not a mature-library CPU speed contest.
Resume wider evaluation only for a specific product correctness/quality/cost
question recorded at reflection, without altering formal budgets.

Current evidence: twelve CPU recipes, installed D1–D5 development checks,
sequential M=1/8/32 semantics and 1264 passing CPU tests (one Linux-only skip).
The paired shared A6 fit improved from 992 to 757 seconds with exact artifacts;
six real comparator probes passed in seconds. These separate observations expose
a practical CPU runtime concern, not a matched-quality speed ratio. Neither test
counts nor adapter counts prove author benefit, quality, device cost or adoption.
No independent author attempt or full search has launched. Guest RSS, address
limits and requested container capacity remain distinct.

The [064–084 roadmap](v1-sprints/roadmap-after-063.md) retains remaining scope;
[070 readiness](v1-sprints/070-readiness-inventory.md) retains open evaluation gaps.
Reflect after each 085 slice, every three implementation commits, phase transitions
or architectural/correctness counterexamples. Keep sprint evidence and learnings
current. Silent correctness failures and demonstrated consumer blockers take priority.

1. Explicit algorithm tasks, fair baselines, and independent correctness oracles.
2. A minimal CPU foundation tested by structurally different use cases.
3. Evidence that agents can make verified algorithm changes with less work.
4. Verified single-GPU execution and scoped end-to-end cost, including train-many.
5. Real use-case value and independent authors' repeated use.
6. Stabilize packaging and the public contracts justified by that evidence.

Silent correctness and persistence failures on any exercised path take priority
within every stage. ScoringBench is a distributional quality instrument, not the
gatekeeper for all foundation work. Follow F0–F5 in the active plan; the previous
P0–P7 checklist is a historical implementation/evidence record.

Treat Ray, multi-GPU, out-of-core training, GOSS speedups, and fused train-many
as experimental. Train-many state semantics are an early design probe; fused
execution and scaling claims require exact correctness and scaling artifacts.
Ray, multi-GPU, and out-of-core expansion remain outside the active plan.
The repository audit in
`learnings/2026-08-15-repository-audit.md` records the current evidence gaps.

## Architecture

The user requested retirement of all old production code during Sprint 002.
`src/openboost/` now implements public CPU data/problem records, immutable run
transactions, named statistics and composable split/routing/leaf operations.
Depthwise, best-first and symmetric growers share these operations across numeric,
missing and categorical features, scalar/vector leaves and routed residual solvers.

Twelve CPU recipes cover squared, binary, multiclass, ranking, quantile, Poisson,
Gamma, fixed-power Tweedie, fixed-scale event/right-censored log-normal AFT,
Normal, saturation Formula and multi-output squared. Normal ordinary/Fisher and
Formula full-GGN updates currently commit jointly. Frequency-severity composition,
class metadata, AFT scale and multi-output inverse scaling have persisted inference
artifacts. PreparedData explicitly reuses fitted training binning/codes across
independent heterogeneous runs. All recipes support independent validation patience
through public StopState, separate from model acceptance and best-model selection.
Training execution is sequential CPU; experimental CUDA storage and aggregation
plus public split/feasibility/route/leaf operations are verified at `9ce790e`.
The separate experimental resident scalar recipe executes on T4 but fails the
088 weighted/missing parity gate. Other required CUDA recipes remain unimplemented.
All A1–A13 real evaluations and formal author/quality/cost gates remain open.

The user approved B03–B06 construction overlapping unfinished F0.3; see Sprint 018
and the active plan amendment. Later CPU slices do not establish a formal phase
exit. The last full old implementation is Git revision `50acfc6`; historical
tests/examples require that revision. There is no compatibility shim or legacy
backend in the current package.

Build the new public data/targets, stats/ops, tree, objectives, runtime, recipes
and artifacts according to the construction design. `tests/v1/reference/` is an
independent mathematical oracle, not the new production backend. The old fixed-bin
sentinel, per-channel trainer and global backend are not new architecture constraints.

## Correctness Rules

- A serialization change requires prediction round trips for numeric,
  categorical, missing-value, and specialized leaf/tree state that it touches.
- A CUDA change requires CPU/CUDA parity for gradients, splits, leaves,
  predictions, and final task metrics—not just matching array shapes.
- Never silently ignore `sample_weight`, exposure, callbacks, evaluation sets,
  constraints, or sampling parameters. Support them or reject them explicitly.
- Randomized behavior must be driven by the model's declared seed; do not use
  unscoped global `numpy.random` state.
- Distributed child histograms must be derived from routed samples. Scaling a
  parent histogram is not an exact substitute.
- Public examples are tests of product behavior. If an example cannot run, fix
  it or label the feature experimental before documenting it.

## Evidence and Benchmark Rules

- Every performance or quality claim must link to a committed raw artifact.
- Record git SHA and dirty state, dataset/version/hash, split seed, package
  versions, OS, CPU/RAM/thread count, GPU/driver/CUDA, and exact CLI arguments.
- Compare end-to-end fit and prediction, including distribution gradients,
  transfers, compilation policy, and fallbacks. Kernel microbenchmarks cannot
  support an end-to-end product claim.
- Use repeated folds/seeds and publish failures. Compare at matched predictive
  quality; do not declare a speed win when CRPS/NLL/calibration regresses.
- Keep official ScoringBench results separate from OpenBoost's large-sample
  extension. See `benchmarks/scoringbench/README.md`.
- Synthetic experiments generate hypotheses. Real third-party datasets and
  upstream-accepted results generate evidence.

## Commands

Use `uv`; do not mutate the project environment with ad-hoc `pip` or Conda
commands.
Current default discovery runs only `tests/v1/`; retained historical tests are
excluded, not counted as passing/skipped v1 coverage. See `tests/README.md`.

```bash
# Install
uv sync --extra dev
uv sync --extra cuda

# Focused test while iterating
OPENBOOST_BACKEND=cpu uv run pytest tests/test_file.py -n 0 -q

# CPU regression suite
OPENBOOST_BACKEND=cpu uv run pytest tests/ -m "not gpu and not benchmark" --tb=short

# Lint production code and changed support files
uv run ruff check src/openboost/ path/to/changed_file.py

# Documentation and packaging
uv run mkdocs build
uv build
```

Run CUDA tests only on real CUDA hardware. A skipped GPU job is not a passing
GPU validation. ScoringBench has a separate Linux environment documented under
`benchmarks/scoringbench/`.

## Working and Commit Discipline

- Use English for all repository prose, including documentation, comments,
  instructions, and new sprint/learning records. Preserve literal dataset values,
  identifiers, formulas, and raw evidence when translating existing prose.

- Keep changes small and cohesive. Prefer root-cause fixes over compatibility
  shims that conceal invalid state.
- Commit after each independently verified slice: test/benchmark harness,
  correctness fix, documentation/learning update, or infrastructure change.
- Do not bundle unrelated cleanup into a fix. Do not amend or rewrite existing
  commits unless the user explicitly asks.
- Do not push, publish, create a release, or update an external leaderboard
  unless the user asks for that external action.
- Before every commit: inspect the staged diff, run the narrowest meaningful
  tests, and include the verification in the relevant learning entry.

## Learning Log

`learnings/` is the durable project memory for decisions, failed attempts,
experiments, and non-obvious operational facts.

- Add or update an entry for every non-trivial change.
- Use `learnings/TEMPLATE.md`.
- Record evidence and falsified hypotheses, not a diary of shell commands.
- Link files and commits. State what was not verified.
- Never include credentials, tokens, private URLs, or user-specific secrets.
- Release notes describe user-facing changes; learning entries explain why the
  implementation and evidence changed.

## Definition of Done

A change is done only when:

1. The intended behavior is covered by a focused test or reproducible artifact.
2. Relevant regression tests and lint pass.
3. Documentation and capability claims match the implemented boundary.
4. A learning entry captures important decisions, failures, and follow-ups.
5. The change is committed as a cohesive unit and the remaining work is stated.
