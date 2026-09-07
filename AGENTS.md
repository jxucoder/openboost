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

Current execution map: [Sprint 038 goal/progress review and plan](v1-sprints/038-goal-progress-and-plan.md).
Installed public D2/D3 development wheels pass in Sprint 040 without core edits.
Sprint 041 adds installed ordered Normal/Formula updates through public transactions.
Sprint 042 resolves external result interoperability through the structural
RecipeResult contract and installed mixed M=1/8/32 checks. Sprint 043 adds an installed D1 expectile objective. Next: remaining
D5 author probes, plus remaining OpenBoost real-data adapters. Sprint 044 connects
current A1/A11 workers to all five frozen housing folds; this is validation plumbing,
not real-data quality acceptance. Sprint 045 adds A6 frozen target-scale binding
and original-unit prediction on all five Parkinsons folds. Next: A13 selection
and remaining adapters/D5 checks. Sprint 046 adds training-scale-verified A6
selection and a synthetic 16-trial current search/release check. Real searches
and remaining adapters/D5 checks are open. Sprint 047 adds verified-scale A6
standardized quality reporting alongside every per-target gate. Sprint 048 adds
A2/A3 probability adapters and five-fold Adult integration; full Covertype runs
and remaining adapters/searches/D5 are open. Sprint 049 records all five full
Covertype folds timing out at the 90-second fit cap. Next profile the current
CPU path on that same input before expanding adapters; A3 validation is incomplete.
Sprint 050 identifies histogram aggregation and repeated candidate row hashing
in a bounded full-input profile. Next hoist invariant candidate row hashing with
exact identity/candidate checks, then rerun the bounded workload. Sprint 051
completes that change: fold zero passes in 87.4 seconds with exact fresh replay;
the other folds were pending at that revision. Sprint 052 reuses selected
histogram statistics with exact conformance checks;
all five full Covertype folds pass within the unchanged cap and replay exactly.
Sprint 053 connects A5 independent quantiles to all five frozen Bike origins
with exact fresh replay and independently recomputed pinball scores. Sprint 054
adds A7 explicit count/exposure binding on all five frozen frequency folds with
exact replay. Sprint 055 adds A8 claim severity on all five frozen grouped folds.
Next: A9 aggregate/composition, remaining application adapters/searches and D5 checks.
These internal trials
do not establish E5/E7.
Independent validation stopping is implemented in Sprint 039. A6 CPU workflows and shared
preparation/M=1/8/32 fixed-budget equivalence are implemented (Sprints 036–037).
Do not infer F1/B11 readiness from implemented objective count. The
[Sprint 035 audit](v1-sprints/035-cpu-coverage-audit.md) remains historical evidence.

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
Execution is sequential; CUDA execution is not implemented.
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
