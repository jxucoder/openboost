# OpenBoost v1: Acceptance and evaluation protocol

Version: v1-plan-r2 / 2026-09-05. Status: preregistered design thresholds; new v1 evaluation not run.
Scope revision: the user requires every application; individual A1–A13 replaces eight representative
tasks. Use with the [main plan](agent-boosting-foundation-plan.md). Numbers are proposed acceptance
criteria, not results. F0 freezes concrete data, implementations and resources before execution.
Postmeasurement threshold changes require a new protocol version; retain old failures.

## 1. Result states and deliverables

Every case records not_run/pass/fail/unsupported/error/timeout. Any non-pass required cell prevents
its gate passing. Optional unsupported records support boundary statements only. No hardware means
not_run, not GPU pass. A missing expected cell fails.

F0 implements benchmarks/v1 manifests, runner/judge. Sprint 011 provides
[artifact integrity](../benchmarks/v1/README.md) for declared matrices/cache identities/files only.
Real freezes, runner and independent quality/E-gate judging remain incomplete; integrity_pass is not a gate pass.
Each immutable run directory includes at least:

- manifest.json: protocol hash, code SHA/dirty, data version/hash/row/split IDs, target/preprocessing,
  versions/wheel/build hashes, OS/CPU/RAM/threads, GPU/driver/CUDA, budgets, exact CLI, seeds,
  A-ID→recipe/test/artifact mapping and expected task×model×fold×device matrix.
- cases.jsonl: cell status, effective parameters, backend/fallback, timing scope, prediction/model hashes,
  metrics/failure reasons, not averages alone. Agent cases also record prompt/tool/model/budget hashes.
- report.json/README.md: gates, comparisons, worst cases, confidence intervals and unverified scope.
- Independent evaluators, complete stdout/stderr/JUnit or equivalent; verify parent-run hashes when referenced.

First test judges with missing folds, changed-config stale caches, NaN, wrong backend, timeouts, skipped
GPUs, nonzero workers, duplicate cases and missing raw predictions. Training code cannot grade itself.
Cache keys include code, data splits, preprocessing, all configuration and protocol.

## 2. E0: Coverage and interface semantics

Required scope: R1–R9/C1–C7/A1–A13. Every item has interface, CPU/CUDA status, failure semantics, test
entry and artifact links: 100% recorded and 100% of required cells passing. Every A-ID appears in
manifest/report with a real task, implementation and independent check. Avoid meaningless Cartesian
products, but never substitute a few applications. Required cells cannot be deleted after results.
Capability support and task quality are separate and cannot offset each other.

## 3. E1: Mathematical, state and persistence correctness

| Category | Independent reference/counterexample | Default acceptance |
|---|---|---|
| Objective/geometry | Float64 formulas, finite differences, small solves; separate loss/gradient/effective curvature | Smooth well-conditioned fixtures rtol=1e-6, atol=1e-8; nonsmooth declared subgradient/leaf optimality, not inappropriate finite differences |
| Histogram/route/split | Original-row reductions, exhaustive candidates, missing/categories/empty/zero-weight/extra stats/ties | Integer counts/routing exact; CPU rtol=1e-7, atol=1e-9; declared tie policy |
| Leaves/topology | Weighted Newton/quantile, scalar/vector, three policies | Oracle match, at least two rounds and propagated state changes, not shapes alone |
| Run/update | Joint/ordered, backtracking/rejection, retry, early stop, RNG, M=1/8/32 permutations | Compare stable IDs; no rejection residue; reordering/grouping preserves seed semantics; retain failures |
| Persistence | Numeric/missing/categories/vectors/coefficients, output schema, offset/link, formula dependencies | New-process prediction roundtrip; corrupt shapes/versions/indices fail; old formats may be rejected |
| Data semantics | Query/entity splits, mappings, unseen, bounds, offset/weights | No cross-split learning, misalignment, double weights or silent fields; valid censoring infinity distinct from NaN |

Deterministic CPU references rerun exactly under the same platform, threads, versions and seed.
GPU reductions need not be bitwise equal: float32 intermediates/default final raw use rtol=1e-4,
atol=1e-5; task metric difference<=1e-3*max(1, abs(CPU metric)). Extreme fixtures preregister separate
stable-reference tolerances; do not globally loosen them after failures. GPU topology may differ
within a predefined tie band if both candidates are within optimality tolerance and final quality
passes. Similar final scores cannot excuse wrong non-tied splits.

## 4. E2: Algorithm expressiveness

Runnable R1–R9 reference recipes use installed public v1 interfaces, no private imports. Verify:

1. Replacement loss/parameter geometry, including an incumbent-friendly control.
2. Custom candidate feasibility/scoring/growth while reusing histogram/routing.
3. Leaf replacement, e.g. weighted residual quantiles, with routed rows/statistics available.
4. Multiparameter adaptive acceptance/rejection and changed parameter order.
5. Shared prepared data across runs with independent RNG, early stopping, errors/results.

All five changes need mathematical/state checks, no core edits/private imports/task-name runner
branches. Independent package installation passes and replacements actually execute. This proves
expression, not new-algorithm quality or easier use than incumbents.

## 5. E3: Real task quality and scope

### Data coverage and splits

F0 fixes a real task/acceptance entry for **each A1–A13**: regression, binary, multiclass, ranking,
quantiles, multioutput, Poisson, Gamma, Tweedie/composed total loss, survival/AFT, NaturalBoost, Formula
and train-many selection. A1–A12 each have predictive quality; A13 has real selection and full-set E4 cost.
Poisson cannot substitute for Gamma/Tweedie; ordinary regression cannot substitute for distributions/Formula.

At least 6 independent sources total. Different targets on one dataset still count as one source;
relabeling/recounting predictions is not another use case. A13 reuse adds no source. Formula requires
synthetic identification/misspecification plus real quality. Before selecting real data, it remains
incomplete, not deferred. Mathematical simulation is E1, never real-task evidence.

Sources come from [application contracts](foundation-application-contracts.md). F0 verifies downloads,
licenses/versions and pins IDs/raw hashes. Unavailable data stays unresolved; replication cannot invent
scale. Sources may change before evaluation, but all applications remain required; unsupported/not_run
cannot pass the application or E3.

General splits: seeds 0–4,60/20/20 train/validation/test, using stratified/group/time logic as appropriate.
Official fixed splits take precedence. Ranking never crosses queries; duplicate entities never cross
splits; temporal tasks use declared rolling origins. Five runs need not be IID. Final test cannot
select parameters, budgets, thresholds or case exclusions.

### Fair baselines

Candidate versions: XGBoost 3.4.1, CatBoost 1.2.10, LightGBM 4.7.0; see [release review](boosting-release-review-2026-09-05.md).
Choose actual supported objectives per task, plus suitable GLM/AFT, NGBoost, Py-Boost or simple structural
baselines. Unsupported is not a worse score; do not require unsupported CatBoost SurvivalAft GPU.

Retain explicit effective defaults. Main comparison uses **16 preregistered configurations per method**;
select configuration and baseline method on validation, then unlock test. Define reasonable per-method
spaces, not equal depth for symmetric/leaf-wise trees. Compare leaf budget/quality when matching size.
Report total model-selection cost. Fixed-trial and fixed-time conclusions are different experiments.
F0 freezes CPU/GPU/time/memory caps per trial; timeout is failure, not free extra retries.

### Standard-recipe quality floor

Compare against the validation-selected opponent, not posthoc best test scores.

| Primary metric | Across-split threshold per task |
|---|---|
| Nonnegative loss: RMSE/log-loss/pinball/CRPS/deviance/Brier | Candidate/baseline median ratio<=1.05, worst<=1.15 |
| Potentially negative NLL/censored NLL | Align parameterization, units, normalization, constants; median per-row difference<=0.02 nats, worst<=0.10 |
| NDCG@10 | Baseline-minus-candidate median<=0.01, worst<=0.03 |

Ratios only when baseline loss>1e-8; otherwise absolute difference<=1e-8 and mark near-perfect,
excluding from ratio summaries. Never take invalid geometric means of zero/negative values.
These are engineering gates, not universal noninferiority to three libraries. Every A1–A12 passes
its own gate; easy tasks cannot average away failures. A13 selected models pass the original task's
quality gate and selection-leakage checks.

Distribution/survival tasks add proper scores/calibration. Report coverage with width; survival
respects censoring/support, not C-index alone. Metrics do not replace target semantics. Report every
split, paired differences and task-level intervals; small task sets do not establish market-wide superiority.

Custom algorithms are reported separately. Mathematically correct but worse quality is a valid
research result, not a quality win. Foundation research value needs at least one E5 modification-cost
benefit. Algorithm superiority needs a matching real task, frozen protocol and reproducible quality gain.

## 6. E4: GPU, train-many and inference cost

Measure semantic reference, same-algorithm optimized implementation and quality-matched external
baselines. Same machine, threads, data/cache policy: one first-fit plus three warm fits, at least 3 seeds.
No profiler/GPU-memory sampling thread during official timing. Report binning, objectives, transfers,
JIT, training, validation, export/prediction; stage times supplement, not replace end-to-end.

Required workloads: small startup/single-row prediction; medium/large real tasks from two sources,
one>=100k rows; K>1 outputs; M=1/8/32 ensembles. F0 fixes shapes/resources before results. Historical
T4 P7 stays separate with original 1.2 threshold, never rewritten.

Cost gates:

- Required device recipes have no silent fallback and pass all E1 CPU/CUDA checks.
- Two medium/large standard recipes: at matched quality, warm-fit median<=2× fastest qualifying
  external GPU baseline. Either exceeding fails the GPU performance gate.
- Public composition/same-semantics optimized warm-fit ratio<=1.25, without ignoring plugins.
- Train-many: M=1 overhead<=10% versus independent path. Freeze at least one real ensemble and
  measure M=8/32 against sequential shared-preprocessing reference. At least one reduces full-set
  time>=20%; the other is no slower than 10%. Other required ensembles also no slower than 10%.
  Per-run quality/seed/stop agree; memory meets fixed cap. Report binning-reuse-only benefits separately.
- Measure new-process import→load→predict and warm batch/single-row inference. Standard CPU median
  ratio<=2 versus fastest qualifying CPU baseline. Declare custom dependencies. Manifest records
  repeats, batch sizes and timer-overhead correction.

Failure may still yield CPU/experimental work, but not v1 GPU acceptance. Not every research recipe
must share one speed target, yet complete costs are recorded. Exact GPU peak, device-wide sampled
lower bounds and process RSS differ. Missing values stay null with explanations, never zero-as-no-memory.

## 7. E5: Agent evaluation

Choose one development task from each E2 modification type and two tasks excluded from API design.
Each has a card, math/state verifier and resource limit; candidates cannot modify judges. Tasks used
for interface changes become development, not unseen validation.

Compare OpenBoost against **F0's most appropriate existing implementation path**, allowing public
hooks, outer loops or source edits with normal docs/recipes/tools. Include Py-Boost/GPU custom-objective
capabilities in selection rather than weak defaults. NumPy can be a separately reported diagnostic control.

Each task/arm has **3 independent attempts**: 15 development and 6 held-out. Freeze agent model/reasoning,
tools, initial docs, cache/prompts. Each attempt caps at **30 minutes or 20k generated tokens**, whichever
comes first; manifest states token accounting/service measurability. Rotate arm order, no patch leakage.
Before large execution, run one smoke task to verify accounting/cost.

Record time to first correct completion, install/active/blocked time, tokens/tool/compute cost, human
hints, core edits/private imports, failure reasons and independent correctness. Failed attempts count
full budget in time aggregates; also report successful-only time, not success-only selection. Changed
versions/settings require a separate cohort.

Gates: OpenBoost development>=12/15 correct and>=2/3 per task; held-outs>=2/3 each, total>=4/6.
No core edits/private dependencies. At least two deep-change types reduce capped median time>=30%
versus the opponent, with no fewer total correct completions. Equivalent controls need not be wins.
These are small-sample engineering decisions, not significance claims. On failure inspect task/docs/API
boundaries, not merely add hints or restrict opponents.

## 8. E6: Installation, delivery and reproducibility

- Clean CPU and one CUDA environment install wheels without source/private-import dependencies.
  CPU install does not require CUDA; declared Python/OS matrix passes individually.
- Two independent extension packages plus all standard recipes run. Core raw/tree inference works
  after training plugins are removed. Custom formulas/links require declared installed dependencies,
  not automatic arbitrary-closure serialization.
- Docs, signatures, capability tables/tests agree. Major errors identify component/field/device/shape/reason.
- Judges reconstruct gates from committed raw artifacts; runtime failures produce nonzero exit.
- Release content has versions/licenses and no private data. Old-format rejection is allowed;
  new round trips must work. Publishing, push and leaderboards still require user-requested external action.

**Engineering v1 complete: individual A1–A13 evidence and every required E0–E6 gate passing.**
Partial work is a milestone/candidate, not full v1. Reports list gate evidence, not total tests as completion.

## 9. E7: Adoption/impact independent of engineering completion

After authorized contact, at least two independent authors try the package; one implements their
own method and reuses it on another task. Record installation, help, failures, reasons and actual
dependencies. Internal agents/repository authors are not external users. Impact means independent
research reuse, downstream method packages or real decision benefits, not stars/downloads alone.
CPU trials can begin before all GPU gates. Until E7 passes, adoption remains an unverified hypothesis,
even if E0–E6 pass; do not claim an ecosystem or broad product value.
