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

Current execution is governed by the user-directed
[Sprint 101 deferral](v1-sprints/101-defer-author-evaluation.md). Pause agent
friendliness studies, 069 author preparation, 077/F2/E5 trials and their 094–100
accounting/isolation infrastructure. Do not request or run the pending Sprint 100
model test. Resume author evaluation only when the user asks. F2/E5 remains
unpassed but does not block foundation construction, required CUDA recipes,
train-many or scoped quality/cost work. Public composability, installed-extension
checks and mathematical/state/persistence correctness remain active requirements.

The approved [Normal CUDA run 8](v1-sprints/102-normal-cuda-validation.md) executes
once at `469ca0e`: 528/529 revised checks pass, and all 26 expected historical
disagreements match. All 409 declared JSON artifacts are retained; both old false
improvements are correctly rejected and all 2,548 trajectory comparison audits
pass. [Raw evidence](benchmarks/v1/evidence/cuda-comparison-092/README.md) preserves
the failed verdict. One forward recipe fixture expects best prefix ten after a
zero-valued tenth term; strict improvement and independent fixture mathematics
require nine. The expectation is corrected locally with three CPU no-op prefix
checks; production code and the consumed run's frozen sources/verdict remain
unchanged. The run-8 allowance is consumed and its raw verdict remains false.

[Sprint 103](v1-sprints/103-normal-cuda-revalidation.md) then executes the narrow
revalidation at clean `a7173d9`: all fifteen recipe cases pass on real T4, including
the corrected forward case's later prediction/comparison/ownership assertions.
All 46 uploaded sources and eighteen pinned packages match. The
[combined audit](benchmarks/v1/evidence/cuda-recipe-103/README.md) verifies 514 earlier
passes plus fifteen new passes with identical production. Bounded revised coverage
is complete across two runs, not one 529/529 invocation or full Normal conformance.
The retrospective is complete; all nine GPU allowances are consumed.

The user then approves the exact 46-file Modal upload for an
[early performance checkpoint](v1-sprints/104-early-performance-checkpoint.md).
Run 10 executes once at clean `c8f7ebc`: one of six cases passes, with every
declared artifact and source/package identity retained. Squared 10,000 rows
qualifies at 7.044 s CPU versus 2.941 s warm GPU, a 2.395 internal ratio with
near-identical quality. Four other timing pairs and the separate warm profile
remain incomplete after child deadlines. Preserve the
[raw failed verdict and partial timings](benchmarks/v1/evidence/early-performance-104/README.md).
The 100,000-row squared GPU has only two of three required warm fits; it has no
qualified ratio. Normal 1,000/10,000-row GPU fits complete, but CPU pairs time out.
All ten allowances are consumed. Exact generated inputs do not reproduce across
the audited macOS/Linux hosts, despite matching same-host CPU/GPU identities;
the next packet must retain lossless input snapshots. Formal E4 remains unpassed.

The user's next "approve" starts
[105 construction](v1-sprints/105-parallel-validation-and-reproducible-cost.md), targeting
parallel boolean/domain validation and complete reproducible cost evidence,
preserving numerical decisions and all public checks. Implement exact input and
per-fit evidence first, parallel field validation second, then freeze a feasible
paired correctness/cost run. New hardware/upload requires that concrete allowance.
Those construction slices are now complete: 1994 CPU checks pass with one
Linux-only skip, original installed CPU replay passes, and the
[88-file / 474-case run-11 packet](v1-sprints/105-validation-run11-request.md)
executes once at clean `dd84247`. All 474 T4 checks and three frozen cost gates pass.
All 28 fits replay exactly from retained input bytes, with original/candidate
models and predictions unchanged. Large squared warm fit cost falls from 13.513
to 8.947 seconds (33.79% lower); smaller squared and Normal cases also improve.
The [raw evidence and offline audit](benchmarks/v1/evidence/parallel-validation-105/README.md)
verify all sources/packages and seventeen retained JSON artifacts. This is a
synthetic internal comparison, not full Normal conformance or formal E4. All eleven
GPU allowances are consumed. Keep the field-validation optimization and stop at
the completed 105 retrospective. Next construction returns to 080 required
objective operations and inference metadata; row validation remains unchanged.

After this checkpoint, advance required CUDA recipes (080), compatible
train-many (081) and evidence-led quality/cost work (082). Preserve all frozen
sources and past results. All R/C/A requirements remain; author/adoption benefit
is unverified. The 105 cost response does not replace any required recipe or gate.

The historical sequence below records prior instructions and evidence. Its
authoring continuations and limited F2-to-F3 overlap rule are superseded by 101;
its statements of pending run-8 authorization predate the completed run above.
The earlier execution was governed by the user-approved
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

The separately approved [096 CPU worker smoke](v1-sprints/096-linux-worker-result.md)
has consumed its one allowance: 14/19 checks pass at clean `7985645`, but the
worker runs as root because Modal ignores Dockerfile `USER`. Core/material writes
succeed, so isolation fails; timeout is not reached. The real evaluator stays
outside the worker and unchanged. Preserve the failed evidence and consumed
freeze. Next local correction is explicit process privilege reduction before
every author command, followed by a separate concrete freeze/allowance. No retry,
model attempt or additional CPU upload is authorized by this result. Independent
token enforcement, fair arms/model/settings and GPU run 8 remain open.

The separately approved [097 corrected smoke](v1-sprints/097-worker-identity-result.md)
passes at clean `518eccf`: all nineteen original checks, both real/effective/saved
UID/GID guards and actual provider expiry. All thirteen original uploaded files
are unchanged; only the trusted launcher is added. Core/material writes and root
restoration are denied; evaluator hashes remain unchanged. Its one CPU allowance
is consumed with no retry. Stop at the recorded retrospective, then return to
actual generated-token/wall-budget enforcement and fair-arm/model/settings design.
No model dispatch, further remote invocation or GPU allowance follows from this
known-code isolation result. The raw failure and correction are both retained.

The user's subsequent "continue" starts [098 request accounting](v1-sprints/098-author-request-accounting.md).
Local text-request construction reserves the remaining cap before dispatch,
counts final output usage once and fails closed on unknown/interrupted usage.
It does not establish real provider token enforcement, cancellation or a complete
author runner. Model tools/worker integration, attempt authority and the concrete
model/input/spend freeze remain open. No model request or remote allowance is
included; injected protocol tests cannot pass the real accounting gate.

[099 background accounting](v1-sprints/099-background-accounting-smoke.md) now adds
bounded cancellation/retrieval and withholding of stopped answers. Its concrete
[model smoke](v1-sprints/099-accounting-smoke.json) was then approved and executed
at clean `5c0f31a`: [cap exhaustion passes, cancellation is unexercised](v1-sprints/099-accounting-result.md).
Actual output is 128 then 64 reasoning tokens and further dispatch is blocked.
The cancellation probe completes early with 104 output tokens; no cancel occurs,
so the frozen overall verdict is failure. Three creates and seven retrievals
produce 296 output tokens, including 275 reasoning tokens. All original artifacts
are retained and the one allowance is consumed, with no retry. Stop at the recorded
retrospective. Next local design should trigger cancellation on an observed active
response; it still requires a new concrete live allowance. Full accounting, worker
integration, independent authors and GPU run 8 remain open.

The user's next "continue" starts [100 active cancellation](v1-sprints/100-active-cancellation.md).
The trusted controller now supports stop on the first validated in-progress
response before the deadline. The recorded 099 prefix exercises that path with
synthetic cancellation replies; all 66 focused checks pass. Its separate
[one-request live packet](v1-sprints/100-cancellation-smoke.json) is pending:
same model/prompt/cap/window as 099's cancellation probe, 4096 output tokens,
$0.01 proposed allowance and zero retries. No new live call has occurred. Preserve
the consumed 099 freeze and archived sources as the active implementation advances;
use its archive verifier for historical evidence. Obtain the concrete allowance
before the new live observation, then stop for retrospective.

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
[088 resident scalar training](v1-sprints/088-resident-scalar-training.md) and
[089 score symmetry](v1-sprints/089-cuda-score-symmetry.md) now pass all 212 real
T4 checks at clean `af026ef`. The 14 weighted/missing failures from run 4 are fixed
without changing original cases or tolerances. Measured device summaries and PTX
reproduce the archived scorer's one-ULP asymmetry; independently rounded products
restore equal scores and the expected split. All 202 original cases and ten added
diagnostics pass. See the [run-5 evidence and retrospective](benchmarks/v1/evidence/cuda-score-symmetry-089/README.md).
Bounded scalar geometry, trees, transactions, retention and saved CPU inference
are verified. The original failed run remains immutable. The user subsequently
approved [090's Normal/D2 run 6](v1-sprints/090-normal-run6-request.md), which executes
at clean `4143d18`: **381/383 pass**, with two ordinary ordered depth-zero
backtracking-decision failures. All 212 scalar regressions, 23 Normal operation
checks, 31 recipe checks and twenty installed-D2/fresh-inference checks pass;
mapped runtime passes 94/96. Nineteen saved models replay without CUDA or the
training extension. [Raw evidence and retrospective](benchmarks/v1/evidence/cuda-normal-090/README.md)
retain both failures, all 79 raw artifacts and the separate known split near-tie.

The separately approved [run-7 diagnostics](v1-sprints/091-acceptance-run7-request.md)
execute at clean `80740f2`: **383/385 pass**, retaining the exact two original
failures. Both new observation cases pass; all original results and nineteen saved
model bytes match run 6. [Actual traces and analyses](benchmarks/v1/evidence/cuda-acceptance-091/README.md)
establish false improvement on round zero's mean update: measured NLL falls by
`8.88e-16` while high-precision math at the same stored inputs worsens by `5.90e-18`.
Rounded gradients/Fisher match the reference; float32 reduction cancellation creates
a small leaf. The transaction follows the reported decision and releases ownership
correctly. No acceptance-policy correction or full Normal conformance is claimed.

All seven allowances are consumed; no retry or additional upload is authorized.
[091's retrospective](v1-sprints/091-normal-acceptance-diagnostics.md) is complete.
The user approved local [092 comparison construction](v1-sprints/092-normal-comparison-design.md):
independent loss-change mathematics and explicit numerical resolution, then a public
objective operation and separate training/best/stopping consumers. The independent
[092-A numerical experiment](v1-sprints/092-normal-comparison-mathematics.md) precedes
production changes and adds no device allowance. Its
[106-case local evidence](benchmarks/v1/evidence/normal-comparison-092/README.md) and
[383-case mapping](v1-sprints/092-comparison-cohorts.md) complete the mathematics gate.
[092-B's public comparison operation](v1-sprints/092-public-comparison-operations.md)
is constructed: the CPU operation passes the 106-case numerical study, and the
resident CUDA operation has a separate 117-case cohort collected but unrun.
092-C's CPU Normal consumers now use separate acceptance/best/stopping comparisons;
1794 CPU checks pass with one Linux-only skip. Resident runs now offer explicit
objective comparison and owned best anchors; twelve new ownership/consumer CUDA
tests collect but remain unrun. The Normal device recipe now selects objective
comparison and owns a separate patience anchor; fifteen additional recipe CUDA
tests collect, unrun. The [collected bindings](v1-sprints/092-cohort-bindings.md)
now cover all 383 requirements, including 147 revised transaction/recipe/D2 cases.
All ninety independent float64 reference trajectories retain their original
summaries. The [pending run-8 request](v1-sprints/092-comparison-run8-request.md)
now freezes 86 files, separately judges 385 historical / 529 revised cases and
collects both from an isolated wheel/snapshot. Thirty-seven harness checks pass;
no run-8 upload or CUDA invocation is authorized. Obtain the concrete allowance,
then execute once and stop for retrospective. Preserve
the old full-loss predicate cases and failed runs; any revised
semantic cohort must be explicit before hardware. No blanket epsilon, hidden metric
adjustment or aggregation-only fix establishes reliable comparison. Original P7/E4
and [069 accounting/isolation](v1-sprints/069-authoring-pilot.md) remain required.
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
sequential M=1/8/32 semantics and 1949 passing CPU tests (one Linux-only skip).
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

1. Resolve demonstrated correctness failures using independent mathematical oracles.
2. Verify programmable single-GPU execution against the CPU semantic reference.
3. Complete required CUDA recipes and compatible train-many semantics.
4. Measure real use-case quality and complete execution cost with fair baselines.
5. Stabilize packaging and public contracts supported by that evidence.

Comparative agent authoring and adoption studies are deferred under 101. Existing
E5/E7 criteria remain unpassed and are not engineering entry prerequisites during
this deferral.

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
CPU recipes execute sequentially. Experimental CUDA storage, aggregation and
public split/feasibility/route/leaf operations plus the separate resident squared
recipe pass the bounded 212-case T4 matrix at `af026ef`, including weighted/missing
parity, owned transactions and saved CPU inference. Normal K=2 device operations,
mapped runtime and joint/ordered recipes execute in the bounded run-6 matrix,
which passes 381/383 checks with two acceptance failures. Run 7 repeats both
failures and captures the false-improvement mechanism; its two diagnostics pass.
Installed D2 and nineteen CPU replays pass; full Normal conformance remains open. Other
required CUDA recipes remain unimplemented.
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
