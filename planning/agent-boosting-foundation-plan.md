# OpenBoost v1: Programmable boosting foundation design, execution and acceptance

Date: 2026-09-05. Version: **the real v1 planning baseline**, explicitly designated by the user.
Status reviewed at merged PR #24 / Sprint 063: **F0.1/F0.2 delivered; F0.3 open; broad CPU construction delivered, formal F1 exit and author/quality/cost evaluation incomplete**.
v1 denotes this product/architecture goal, not the existing PyPI version or a claim that P0–P7 completed v1.
Code-review baseline: 3ac1552; branch: codex/gpu-python-foundation-design.
Execution: [v1-sprints](../v1-sprints/README.md); Sprint 010 completed the
[F0.2 exit audit](../v1-sprints/f0-2-acceptance-ledger.md). Each sprint records plan/results/reflection;
phase status follows evidence. At the user's request, Sprint 002 retired old production early.
Reproduce it at `50acfc6`. The package now has shared CPU components, twelve
recipes and explicit training-preparation reuse. See the
[current retrospective and execution plan](../v1-sprints/063-retrospective-and-next-plan.md)
for delivered boundaries and remaining work, and the
[064–084 sprint roadmap](../v1-sprints/roadmap-after-063.md) for bounded execution cards.
Amended phase dependencies and
acceptance gates below still apply; implemented recipe count is not a phase exit.
The user-approved 2026-09-07 amendment adopts bounded scalar CUDA feasibility
alongside exploratory authoring; see [085](../v1-sprints/085-foundation-focus-amendment.md).
It supersedes the earlier unadopted-overlap proposal. Formal acceptance remains
unchanged. Historical checkboxes require current evidence reconciliation.
The [086 execution plan](../v1-sprints/086-next-execution-plan.md) specifies the
next fields/histogram slice, its frozen fixtures, the remaining device allowance
and the path to resident training. Storage is verified; CUDA boosting remains open.

This plan supersedes future investment order in the old GPU checklist and the
[ScoringBench-first plan](impact-adoption-value-next.md). Old experiments and failures remain valid.
The user states that the foundation is the product and use cases determine abstractions.
**No backward compatibility is required: APIs, trainers, representations and persistence may be rewritten.**
Planning/F0.1 did not start a rewrite or GPU job; subsequent execution follows these dependencies.

The v1 specification consists of:

- This file: product goals, scope, architecture, dependencies, phase deliverables and completion.
- [Application contracts](foundation-application-contracts.md): all required A1–A13 and data choices.
- [F0.1 task cards](foundation-tasks.md): inputs/outputs, data, mathematics, oracles, baselines, interface sketches.
- [Construction design](foundation-construction-design.md): modules, records, function contracts,
  tree/state/device execution and B01–B14 implementation slices.
- [Acceptance/evaluation](openboost-v1-evaluation.md): E0–E7, quantitative gates, comparisons, failures and artifacts.
- [Release/plan review](boosting-release-review-2026-09-05.md): sourced competitive facts and shipped/experimental/planned status.

**Every listed case requires individual delivery and acceptance.** Classification, regression, ranking,
quantiles, multioutput, counts, positive amounts, total loss, survival, distributional prediction, Formula
and train-many jointly constrain the foundation. Insurance/AFT have no special priority. A few
important cases or a task count cannot replace A1–A13. Dependencies determine implementation order,
not whether a case belongs in v1 completion.

## 1. Revisit the product hypothesis

Help researchers and agents turn a boosting idea into a verified, reusable implementation at lower
cost. Optimize **total cost from algorithm change to trustworthy results**: implementation, debugging,
verification, repeated training and deployment, not just one fit, Python percentage or public-function count.

Validate four separate hypotheses rather than merging them into one technical test:

1. Useful problems require internal algorithm changes that existing configuration/hooks cannot satisfy cheaply.
2. Composable components substantially reduce that work across structurally different algorithms.
3. Agents can use them correctly and receive enough diagnostics to locate failures.
4. End-to-end workflow benefits outweigh runtime overhead, learning cost and a new dependency.

Initial users are researchers, domain modelers and their agents with actual modification needs.
One-call tabular recipes help trials; independently publishable author recipes/packages are the
adoption path. Users enter through a method, then reuse components and publish their own.
Impact means reuse in independent research/decisions; adoption means independent completion/repeated use;
value means total verified-change cost and task results. Do not yet infer willingness to pay or a business model.

### Correct competitive assumptions

- XGBoost has Python custom objectives/metrics, grow policies, updaters and leaf constraints.
  Custom loss, depthwise/lossguide switching or simple leaf clipping is not automatically distinctive.
  [Custom objectives](https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html),
  [parameters](https://xgboost.readthedocs.io/en/stable/parameter.html).
- LightGBM exposes update(fobj=...), rollback and set_leaf_output; existing libraries are not entirely
  closed to training/leaf changes. [Booster](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html).
- NGBoost supports new distributions, scores and metrics; Laplace is an easy control task.
  [Development guide](https://stanfordmlgroup.github.io/ngboost/5-dev.html).
- Py-Boost provides Python GPU boosting/customization and is a direct foundation comparator,
  not just XGBoost/LightGBM. [Repository](https://github.com/sb-ai-lab/Py-Boost).
- Review-date stable versions: XGBoost 3.4.1, CatBoost 1.2.10, LightGBM 4.7.0. XGBoost 3.4 expands hist
  vector leaves; CatBoost has GPU custom objectives; LightGBM 4.7 adds GPU/data interoperability.
  See the release review for limits/plans. Do not assume competitors lack multioutput or custom GPU losses.

These are review-date documentation findings; pin actual versions and verify task paths when executing.
The opportunity is modification cost/verifiability, not denying competitor extensibility. Controls
may use hooks, outer loops and source edits; never artificially restrict competitors.

“At least as good as XGBoost” has separate meanings: expressing a standard algorithm, matching task
quality, and speed/engineering completeness. Arbitrary agent changes cannot be guaranteed to improve
results. Supply reproducible baselines, select with validation and evaluate independent test.
Baseline selection protects workflow value, but calling XGBoost is not rebuilding it with our components.

## 2. Extract variation axes from use cases

These are foundation test programs, not four simultaneous complete product lines.

| Use case | Required algorithm decisions | Minimal testable implementation |
|---|---|---|
| XGBoost/LightGBM-style GBDT | Statistics, split scoring, leaf solving, depthwise/best-first, sampling | Scalar second-order boosting on fixed bins; two growth orders sharing statistics/routing |
| NaturalBoost/NGBoost-style | Links, proper score, Fisher/curvature, direction, learner fit, step acceptance | Two-parameter Normal; fixed steps/bounded training-loss backtracking; rejection cannot pollute model |
| FormulaBoost | Separate theta(Z) from formula(theta, x), parameter coupling, structural inputs/outputs | Two-parameter formula, independent Jacobian/GGN, at least two rounds composed from trees |
| Train-many | Data reuse, independent state/RNG/stopping, execution groups | Multiple recipes/configurations on shared data; sequential reference invariant to run order |

At the reviewed historical baseline, FormulaObjective already solves coupled GGN directions then
fits per-parameter trees. This differs from shared topology/joint leaf solving; do not claim the
former was inexpressible. fit_trees_batch already had a sequential shared-binned-input path;
retain its semantics, not count renaming/wrapping as train-many progress.

Keep two tasks out of interface design to detect overfitting to our examples. Synthetic data may
check mathematics initially; adoption/value require real scenarios.

**Applications** form another axis: classification, regression, ranking, quantiles, counts/positive
values, censoring, multioutput, structure and model selection. Insurance/AFT probe offsets/units and
censoring; ranking probes within-group dependencies; classification probes links/mappings; multioutput
probes shared trees/parameter axes. Structure×application is a design matrix, not a mandatory first-round
Cartesian product. Every application still needs implementation, independent semantics and real evaluation.
Dataset choice cannot make applications optional. A1–A13 plus R1–R9/C1–C7 define v1 scope.

### Minimum v1 deliverables

All capabilities below are required at v1 completion. F1 can begin with a subset but cannot call
it complete v1. CPU is both independent reference and execution entry for all recipes; GPU has explicit subsets.

| ID | Required recipe/task | Minimum behavior/check | v1 CUDA scope |
|---|---|---|---|
| R1 | Regression, binary, multiclass | Squared/logistic/softmax, mappings, weights, probabilities, missingness | Numeric dense including missing required; native categories later |
| R2 | Quantile/robust | Weighted quantile accessing routed residuals/weights, not just G/H | CPU required; CUDA explicitly optional |
| R3 | Group ranking | Pairwise logistic and NDCG-weighted lambda, groups/pairs/group metrics | CPU required; CUDA optional |
| R4 | Count/positive/aggregate | Poisson offset, Gamma/Tweedie fixed-parameter means; targets separate from weights/exposure | Poisson offset required; Gamma/Tweedie optional |
| R5 | Censored AFT | Fixed noise family/scale, events/right censoring, canonical intervals; reject others | Numeric events/right censoring required |
| R6 | Natural/distributional | Two-parameter Normal, Fisher/ordinary, fixed/backtracking, proper scores | Numeric Normal required |
| R7 | FormulaBoost | Two-parameter formula, links, structural inputs, independent Jacobian/directions, two rounds | CPU required; mixed execution declared separately |
| R8 | Multioutput/vector leaves | Independent trees and shared vector topology; split/leaf statistics may differ | Numeric squared-error/diagonal statistics required |
| R9 | Train-many | M=1/8/32, shared prepared data, independent config/seed/stop/failure, sequential reference | At least one batched compatible R1/R4/R8 group required |

| ID | Cross-recipe capability | Completion standard |
|---|---|---|
| C1 | Typed data/targets | Numeric/missing, CPU categories, mapping/unseen, weights, offset, group, interval, structure; train-only preprocessing |
| C2 | Composable trees | Additive stats+one extra channel, replaceable feasibility/score, three CPU policies, actual routed rows |
| C3 | Learner/leaf replacement | Scalar/vector payload, Newton/weighted quantile, output schema separate from parameter axis, custom split constraint |
| C4 | Explicit runtime | Scoped device/workspace/RNG, candidate/accept/reject/stop, independent runs, visible sync/transfer/fallback, no global backend |
| C5 | Artifacts | Raw/user spaces, mappings, base/link/offset, trees/coefficients/vector leaves, new persistence, readable diagnostics |
| C6 | Evaluation/authoring | Independent oracles, five change types, two unseen tasks, real tasks, all failures, external wheel packages |
| C7 | Usable workflow | Install→baseline→read/change recipe→verify→save/load→CPU inference; capabilities/errors/reproducibility |

Native categorical v1 may select one verifiable split method; full CatBoost ordered boosting/CTR
replication is not required. CSR/CSC, text/embeddings, full Cox/competing risks/truncated survival,
all constraints, linear leaves, DART/GOSS, distribution catalogs, autodiff compilers, Ray, multi-GPU and
out-of-core belong to a justified later list. Never silently accept their parameters. F0 records
competitor status and why v1 excludes them while preserving extensible boundaries.

## 3. Architecture: Programmable algorithms, explicit state and bulk operations

This is the overview; module/function decisions live in [construction design](foundation-construction-design.md).
Planning includes how to build the foundation; tasks/evaluation constrain and judge it. F0.2 is
independent mathematical preparation; public components, recipes and runtime are built in F1.

The old Booster(objective, builder, schedule) shell need not survive. **Recipes are ordinary Python
algorithm programs and may own the loop. Runtime manages resources/state without imposing one
training order on every algorithm.** Convenience models are thin recipe entry points.

| Boundary | Responsibilities | Do not hardcode |
|---|---|---|
| Problem/parameter state | Data roles, targets, weights, structure, raw parameters, links, initialization/evaluation | One y for every task; all auxiliary data are weights |
| Recipe | Scores/directions, growth, learner choice, parameter order, acceptance/stopping | One tree per parameter each round; only predetermined learning-rate schedules |
| Components/bulk ops | Binning, routed aggregation, candidate stats/scoring, partition, leaf solves, predict/update | Only G/H, fixed511-slot trees, loss equals curvature |
| Runtime/run state | Device/stream/workspace, buffer ownership, RNG, commit, batch execution | Global backend, parameter axis equals model axis, automatic arbitrary-Python compilation |
| Artifacts/verification | Serializable model state, declared inference, diagnostics, independent references | Arbitrary closure serialization; verification proves arbitrary algorithm correctness |

Prefer a few arrays, structured records and ordinary functions. Initial choices may change with
actual F1 usage; freeze public protocols only after multiple consumers validate them.

### How much of the algorithm is programmable

Design pseudocode, not an existing or finalized API:

```python
with runtime.run(data, seed=seed) as run:
    state = initialize(problem, run)
    for step in range(budget):
        geometry = problem.geometry(state)          # explicit loss/derivative/curvature semantics
        targets = direction_rule(geometry, state)   # may couple parameters
        learner = grow(data, targets, split_rule, leaf_rule, growth_policy, run)
        candidate = propose(state, learner)         # accepted state is unchanged
        decision = accept(candidate, problem, run)  # fixed step or backtracking
        state = run.commit(state, candidate, decision)
    artifact = export_model(state)
```

grow must itself be readable composition: aggregate→candidate statistics→score/feasibility→
choose→partition→solve leaves. Authors can replace a piece or call operations to write another
grow. Do not expose grow only to hide decisions in a second black box. A callback/registry or
universal scheduling graph for every small function is unnecessary.

### Required semantics

- **Statistics versus directions:** name exact Hessians, PSD approximations, Fisher, pseudo-responses
  and fit weights separately. Weight application has one visible owner; do not disguise all directions as G/H.
- **Proposed versus accepted state:** joint/ordered behavior is explicit; backtracking has bounded
  trials. Rejection leaves no trees, coefficients, raw or best-state residue. RNG follows run/step/component,
  not scheduler order. Full data/model copying per proposal is not required.
- **Extensible shared statistics:** start with G/H/count plus one extra additive channel; declare
  shape, dtype, reduction, memory budget/supported scores. No promise to accelerate arbitrary Python reducers.
- **Leaves beyond additive statistics:** weighted quantiles need routed rows/residuals via explicit
  views or declared statistics; do not force them into Newton form.
- **Cross-row objectives:** ranking pairs/queries and Cox risk sets cannot be excluded by a row-independent
  protocol. Direction computation differs from tree per-row reduction.
- **Model versus parameter axes:** one run has K parameters; M runs may have different K, tree counts,
  configurations and stopping. Begin with independent state lists, packing compatible groups only;
  do not require giant M×N×K tensors.
- **Learners versus parameter updates:** learners declare output schemas; recipes map them to parameters.
  Runtime does not equate one tree with one channel. Separate topology/payload. Scalar first is allowed,
  but R8 must actually verify vector leaves in v1, not merely leave conceptual space.
- **Identity for reuse:** sharing requires matching training rows, bin strategy, binning weights and
  metadata. Do not bin full data across validation folds. Align indices, structure and weights.
- **Explicit inference capability:** core readers predict standard trees/raw parameters; custom
  formulas/links declare dependencies or supported representations. Do not promise inference after
  arbitrary formula code is uninstalled. New formats may reject old ones, but their own round trips must work.

### Python and GPU tradeoffs

Python owns editable decisions; large arrays stay on device. GPU remains a design goal; CPU is an
installable entry/reference and device execution is considered early. Start with NumPy references,
CuPy arrays and a few existing CUDA kernels; change kernel DSL only when profiling justifies it.
Numba/CuPy/Python percentage are not product promises. Do not first build a compiler or compare every GPU framework.

Verified common combinations may use batched/fused implementations with identical semantics to
readable composition. Changing recipes requires capability rechecking, not silently ignoring edits.
Arbitrary Python loops may synchronize or not accelerate; reports must say so. GPU baseline fits
retain device trees/predictions and export explicitly rather than CPU-snapshotting every tree for prediction.

Separate development oracles, runtime boundary checks and export validation. Redesign ownership/check
granularity instead of adding a trust-plugin switch to an inefficient path. Cheap checks do not
replace mathematics; read-only arrays/verifiers are not security sandboxes for arbitrary Python.

## 4. Existing assets and constraints

| Asset/constraint | Treatment |
|---|---|
| Independent CPU math/routing, real CUDA conformance, wheel installs, failure reproductions | Preserve semantics/raw evidence; rewrite tests for new APIs without deleting correctness conditions |
| Channel-based _trainer.py and round/channel/lr schedule | Replace entirely if useful; do not inherit its loop restrictions |
| Exact TreeStructure, depth0–8, fixed slots, host snapshots | Old boundaries, not requirements; new layouts still declare resource/support limits |
| FormulaObjective, distribution mathematics, sequential fit_trees_batch | Reuse verified mathematics or rewrite, never as the sole oracle |
| Old APIs/loaders/experimental plugins | No compatibility, dual-write, deprecation cycle or shim required; historical revisions reproduce old behavior |
| ScoringBench parameter fix `3ac1552` | Preserve useful fix for distributional quality, not whole-foundation scheduling |

[P7 T4 evidence](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md)
shows experimental warm fit **12.888×** same-run legacy CUDA, failing the original 1.2 gate.
This shows real overhead, not that programmable foundations lack value. Do not rewrite the old
threshold; freeze new algorithms/workloads separately before measurement. The independent extension
also worsened a proper score; retain that counterexample.

## 5. Execution order and acceptance

Use F0–F5, not old completed P0–P7 labels. Each subtask is a reviewable commit. F0.1/F0.2 are delivered;
continue F0.3 through dependencies/evidence. Construction-design section9 maps phases to B01–B14.

### F0: Specify required algorithms and judging first

- [x] **F0.1: Task cards and alternatives audit.** [Cards](foundation-tasks.md) fix all recipes'
  inputs/outputs, two-round state changes, boundaries/oracles from four structural probes. Audit
  incumbent hooks and source-edit paths/costs, especially Py-Boost. Complete R1–R9/C1–C7/A1–A13
  with shipped/planned status, data/targets/outputs, oracles, processing, metrics, phases/evidence.
  Missing data remains pending, never removed or covered by another task. Cards/interface sketches
  map requirements to E0–E6 instead of another vague roadmap. Specification coverage passes;
  actual bytes/hashes, capability smoke, budgets/held-out tasks belong to F0.3. Engineering design
  supplies records, ops, three growth policies, transactions, inference, CPU/CUDA/train-many construction;
  specifications are not existing public components.
- [x] **F0.2: Original NumPy references/counterexamples.** tests/v1/reference independently
  enumerates splits, reduces rows and solves small matrices without production imports. Hand
  cases cover duplicate-threshold ties, empty/zero-weight children, invalid candidates, rejection,
  parameter order and exchanged run order. External GBDT comparisons distinguish binning/ties;
  different algorithms need not have bitwise-identical full trees. Cover class/link, query/pair,
  weighted quantiles, vectors, offset/weight/units, event density versus censoring probability,
  distribution geometry, formula Jacobians and isolation. Valid infinite bounds must not be rejected as generic non-finite labels.
- [ ] **F0.3: Freeze/implement comparison protocol.** benchmarks/v1 manifest/runner/judge pin
  library/agent versions, tasks, public/held-out examples, budgets, tolerances, quality thresholds,
  CPU/GPU environments and expected matrix. Run old/reference baselines first; declare cost budgets
  before new code. Implement [E0–E7](openboost-v1-evaluation.md), testing missing/polluted/failed artifacts.
  Pin recent three-library releases; CPU smoke before GPU-variant preflight. Do not inherit stale
  completion labels. No full evaluation before actual budgets/data hashes are frozen.

Initial development probes: an incumbent-friendly new-loss/distribution control; candidate splits
requiring minimum per-cohort information mass, with independent information_weight separate from
training weights; and bounded multiparameter backtracking/rejection. The cohort task is a statistical
stability experiment, not claimed innovation or proven demand. If incumbents solve it easily, record
that rather than increasing difficulty. Leaf solving/run scheduling complete the five E2/E5 change types.

F0 exit: all A1–A13 cards, falsifiable judgments, fair comparators and explicit changes. Missing data
or judges prevent exit; task totals cannot substitute. Complete F0.1–F0.3 before F1 except for the approved B03–B06 overlap below; separate design/
reference commits must not smuggle in model migration or kernel optimization.

### Approved sequencing amendment: 2026-09-06

The user approved the [Sprint 017 audit](../v1-sprints/017-f0-sequencing-audit.md).
B03–B06 CPU architecture construction may proceed while B02/F0.3 remains open.
This changes order only: all R1–R9/C1–C7/A1–A13 scope, thresholds, source and
protocol obligations remain required. No CPU interface freeze, formal agent
comparison or quality/speed/v1-completion claim is permitted through this overlap.
B03 starts with independent ownership, weight/offset, transaction and persistence
checks; B06 must probe Formula and heterogeneous sequential runs before stabilization.
See [Sprint 018](../v1-sprints/018-b03-cpu-state.md) for execution and acceptance.

### F1: CPU foundation, close a small path then all v1 coverage

- [ ] **F1.1: Data/run/artifact semantics.** Explicit device/run identity, unique weight ownership,
  parameter/model axes, propose/commit/reject, RNG, input ownership and versioned models. Use F0 state
  fixtures, not wholesale old-model migration.
- [ ] **F1.2: Composable trees.** Aggregate, candidates, score/feasibility, partition, leaves, predict;
  two growth policies as Python. Start weighted numeric scalar constant leaves, then missing/CPU
  categories. Independent symmetric growth reuses statistics/routing, not three cloned builders.
  Extra information mass flows through the same pipeline, without trainer task-name branches.
- [ ] **F1.3: Two complete small recipes.** Standard second-order GBDT and two-parameter Normal,
  including fixed steps and accept/reject. At least two rounds, intermediates, predictions, new-format save/load.
- [ ] **F1.4: Two early structural probes.** Formula two rounds with structure/links; train-many
  M=1/2 then8 with independent state, different stopping and one failed run. Failure cannot contaminate
  others or disappear from summaries. Reorder stable run IDs to verify invariance.
- [ ] **F1.5: Application data/target contracts.** Connect class/link, query/pair, residual/weight,
  vector outputs, positive-target units/offset, AFT events/censoring and distribution/formula inputs/outputs.
  F0 counterexamples check indices, acceptance, inference/round trips. If generic runner needs task-name
  branches, fix boundaries first. Commit recipes separately; unsupported AFT censoring explicitly fails.
- [ ] **F1.6: Complete CPU coverage.** Binary/multiclass, ranking, weighted quantiles, Gamma/Tweedie
  means and vector leaves in independently verified commits. Cross-process persistence covers
  new state, categories/output/inference. Complete R1–R9/C1–C5 CPU cells and local C6/C7 entry points;
  author comparisons and clean-environment delivery remain F2/F5. Every A1–A13 links to runnable
  CPU recipes/workflows, not merely cards or illustrative demos.

F1 exit: required recipes compose the same semantic components. If Formula/train-many bypass core
state, redesign before freezing interfaces; do not defer them. No full feature/speed parity promise
with XGBoost, LightGBM or NGBoost. F1.5/F1.6 must pass before CPU interface freeze. Broader censoring
and real quality remain separate. Acceptance: CPU E0, E1, E2; F1.1–F1.3 alone is only a minimal architecture
milestone. Exploratory F2.1 may start after F1.3 to catch usability problems early; formal F2.2 waits
for required CPU capabilities and frozen interfaces/judges.

### F2: Agent algorithm-change experiments before stabilizing interfaces

- [ ] **F2.1: Exploratory trials.** Install wheel, perform F0 tasks in independent directories,
  record assistance, core/private imports, failures/fixes. Redesign is allowed; these trials are not scored.
- [ ] **F2.2: Frozen comparisons.** Same agent/tools/budget; OpenBoost versus appropriate incumbent
  hooks and, when needed, source edits. Optional NumPy cost reference. At least three independent
  attempts, rotate order; record setup, active time, tokens/compute, human hints, correctness and code
  understood, not LOC alone.
- [ ] **F2.3: Held-out tasks.** Use tasks excluded from F1 redesign. If used to change interfaces,
  reclassify as development. Candidates cannot overwrite verifiers/final evaluation. Small samples
  are directional evidence, not statistical significance or external adoption.

F2 exit follows E5: five development types, two held-outs, fixed attempts/budgets, correctness and
at least two deep-change cost wins, with control results retained. If advantage is only docs/prompts,
fix docs and rerun fairly. Repeated core edits indicate poor abstraction, not something extra wrappers
automatically solve. Rerun affected evaluation after semantic changes.

### F3: Make proven compositions work on one GPU

- [ ] **F3.1: Device-resident vertical path.** Implement bulk ops under F1 semantics, verifying
  gradients/curvature, stats, splits, leaves, round updates, quality/inference end to end. First reproduce
  P7 with its original threshold; score new recipes under separate F0 protocol.
- [ ] **F3.2: Extensions actually execute on device.** At least one nondefault F2 component and
  multiparameter update run on GPU. Expose fallback/sync. Count dynamic backtracking synchronization;
  do not substitute fixed steps.
- [ ] **F3.3: Compatible train-many groups.** After sequential parity, test M=1/8/32 reuse/batching.
  Group different parameter counts/round budgets; fusion need not support every recipe. Record
  throughput, latency, memory limits, failures, JIT/transfers/end-to-end quality. Single GPU first, no Ray/multi-GPU.

F3 exit: readable/optimized forms of the same algorithm agree and meet frozen workload budgets.
Use authorized Modal for bounded timed runs with full provenance. Failure may retain CPU research
value but cannot support GPU cost claims; do not hide structural overhead by deleting verification.
Complete required R1/R4/R5/R6/R8 CUDA subsets, R9 batching and E4. Kernel speedup cannot replace
performance gates or all synchronization/transfer costs. Optional CUDA recipes list CPU results/device status separately.

### F4: Real use cases and independent adoption

F4 need not wait for all F3. Prepare CPU trial material after F1; authorized external trials may
start after F2 interface revisions. Actual user obstacles may change GPU optimization priorities.

- [ ] Complete E3 for every A1–A13, with raw artifacts and at least 6 independent sources. Evaluate
  Poisson/Gamma/Tweedie separately; proper distribution scores, Formula counterexamples plus real
  tasks, and complete model-selection quality/cost. Use strong baselines, preregistered metrics,
  independent test, repeated seeds/folds and all failures. No representative-only subset or synthetic-only Formula.
- [ ] Supply installable wheels, recipe sources, minimal reproductions, component contracts and
  diagnostic examples. Third-party-defined tasks are more valuable than more self-authored demos.
- [ ] After user-authorized contact, at least two external authors try it; one implements their
  own method and reuses it on another task. Record completion, obstacles/reuse reasons. No current
  external-trial/retention evidence exists.

F4 has separate exits: E3 engineering quality and E7 repeated independent use/reproducible benefits.
Internal agent success is not external adoption. Do not wait for an entire ecosystem/all models/
perfect benchmarks to prepare trials. Unauthorized outreach is not a blocker for other engineering;
E7 must pass before claiming adoption validation.

### F5: Stabilize product boundaries from evidence

- [ ] Stabilize repeatedly used contracts only; organize independent recipe packaging/discovery.
- [ ] Remove unused rewritten paths/duplicate concepts. Migrate correctness cases to new semantics,
  not old import/signature/format compatibility. Historical revisions reproduce old behavior.
- [ ] Verify install→baseline→algorithm change→verification→save/inference and update public docs.
  README cannot advertise new architecture before implementation passes.

### v1 completion and dependencies

Main dependency: F0→F1→F2→F3→F5. F4 baseline preparation starts in F0, formal quality follows stable
recipes, and external trials proceed independently after F2. Phase commits link E-gates, raw records
and incomplete cells. Engineering v1 requires individual A1–A13 evidence and every required E0–E6
pass. E7 is separate for external adoption/impact claims. Unfinished cases stay unfinished, not
renamed later cases to declare completion. Revisit task/component reuse at phase exits, not just function counts.

New code uses public modules/ordinary functions: data/targets, stats/ops, tree, objectives, runtime,
recipes, artifacts/models. See construction design for dependencies. Replace private modules as
needed; do not permanently maintain two trainers. Tests live in tests/v1, evaluation in benchmarks/v1.
Reproduce old baselines from fixed historical wheels/revisions without sacrificing the redesign.

## 6. Execution discipline and stop conditions

No compatibility requirement does not demand indiscriminate rewriting. Architecture may start
fresh; reusing correct computation that fits new boundaries is optional engineering judgment.
No signature-preserving shims and no destruction of counterexamples, evidence, user edits or history
in the name of starting over.

Breaking changes are allowed early; freeze a version for formal F2 comparisons. APIs need not
support arbitrary plugins forever, but every evidence artifact must identify exact versions.

- Track expressiveness, correctness, runtime cost, agent-change cost and adoption separately.
- Control-only advantage may require merely better tutorials/adapters; investigate first.
- If every use needs specialized core code, narrow/redesign reusable boundaries and pause feature expansion.
- If quality improves without less implementation work, record an algorithm result, not foundation value.
- Persistent GPU budget overruns require residency/batch/sync diagnosis, not immediate multi-GPU expansion.
- If evidence cannot justify the next investment, commit facts/revised plan rather than more code as a substitute.

Start each change with the smallest independent failure. Use uv tests/lint; public capability changes
also check docs/package. Update learning, inspect staged diff, commit. Migrate only affected semantics;
removed compatibility interfaces have no ongoing pass obligation. This planning round checks Markdown
links and git diff --check, not nonexistent runtime changes. No push, release, messages or leaderboard submission.

Starting instructions for the next execution model:

> Read AGENTS.md and the approved Sprint 085 amendment. Prioritize 069 D1/D2
> authoring/accounting and bounded 078 scalar CUDA feasibility. Pause the next
> OpenBoost CPU configuration-05 probe and wider search expansion. Carry the D2
> composition onto the programmable device boundary. Audit actual ownership and
> enforce independent accounting/isolation; designer work is not author evidence.
> Preserve all R/C/A/E requirements, E5 cohorts, required CUDA recipes and P7/E4.
> Record results and reflect at each bounded slice. No phase exit follows from
> objective counts, green tests or a single scalar GPU fixture.
> Follow Sprint 086's 078-A fixtures next. One 900-second device run remains;
> stop for a retrospective after that run. Further device runs need a new bound.


### Approved foundation-priority amendment: 2026-09-07

The user approved prioritizing measured algorithm authoring and a bounded GPU
path over further CPU resource qualification. The controlling execution card is
[Sprint 085](../v1-sprints/085-foundation-focus-amendment.md). Pause the next A6
OpenBoost configuration-05 probe and wider CPU search expansion. Existing
correctness/evidence obligations and all R1–R9/C1–C7/A1–A13/E0–E7 scope remain.

Exploratory 069/F2.1 D1 control and D2 deep-change measurement may now overlap
bounded 078/B12/F3.1 scalar device construction before formal F2/E5 completion.
Audit relevant 065/068 ownership on the actual device path. The same D2 composition
must subsequently exercise the programmable device boundary. This is an explicit
exception to the default F2→F3 entry sequence, not an F2/F3/E4/E5 exit or permission
to drop Normal/P7, required device recipes, full quality or held-out evaluation.

CPU is a correct, usable reference/development backend; mature-library CPU speed
parity is not a standalone objective. Measure author benefit and practical execution
of useful compositions. The amendment defines acceptance, bounded device budgets,
reflection points and the conditions for revisiting paused evaluation work.
