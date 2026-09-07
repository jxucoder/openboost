# Sprint 038: Goal, progress and remaining execution plan

Reviewed revision: `8afce35`, clean branch `codex/gpu-python-foundation-design`.
Date: 2026-09-06. Status: review complete. M1 is delivered in
[Sprint 039](039-independent-stopping.md). M2 has installed D2/D3 development
evidence in [Sprint 040](040-installed-extensions.md) and ordered D4 evidence in
[Sprint 041](041-ordered-updates.md). [Sprint 042](042-result-contract.md) closes
external result interoperability. [Sprint 043](043-expectile-extension.md) adds
installed D1 expectile evidence. [Sprint 044](044-current-worker.md) starts M3 with
current A1/A11 validation workers; [Sprint 045](045-multioutput-worker.md) adds A6
scale-bound validation integration. [Sprint 046](046-scale-bound-selection.md)
adds scale-bound selection and a synthetic current search. [Sprint 047](047-multioutput-quality.md)
adds standardized A6 quality reporting. [Sprint 048](048-classification-workers.md)
adds classification adapters and Adult validation. [Sprint 049](049-covertype-worker.md)
records five Covertype worker timeouts: profile this full-data CPU path next,
before additional adapters. [Sprint 050](050-covertype-profile.md) identifies
histogram aggregation and repeated candidate row hashing; next remove invariant
rehashing with exact conformance checks. [Sprint 051](051-candidate-row-hash.md)
implements that change and completes fold zero within the cap; histogram cost
and the remaining full folds are addressed in [Sprint 052](052-histogram-gather.md).
All five bounded replays pass after histogram gather reuse.
[Sprint 053](053-quantile-worker.md) adds current A5 independent quantile
integration on all five Bike origins. [Sprint 054](054-count-worker.md) adds
A7 count/exposure integration on all five frequency folds. Next: A8 severity,
remaining application adapters/searches and D5 checks. Remaining M2 and M3–M6 are open.
This updates execution priorities after Sprints 036–037. It preserves the
[main plan](../planning/agent-boosting-foundation-plan.md),
[construction design](../planning/foundation-construction-design.md), all
R1–R9/C1–C7/A1–A13 requirements and the [E0–E7 gates](../planning/openboost-v1-evaluation.md).

## Goal and assessment

OpenBoost is a **programmable boosting foundation for researchers and agents**.
Its value is reducing the total cost from an algorithm idea to a correct,
reusable implementation with trustworthy task results. Implementation, debugging,
verification, repeated training and inference all count toward that cost.

The torch analogy describes the role of composable operations and explicit
state, not a requirement to build a general tensor framework. Readable Python
algorithm code and efficient device operations are means to this goal. A useful
foundation must allow changes to statistics, split decisions, leaf solvers,
parameter updates and scheduling without repeatedly modifying its core.

Standard GBDT, distributional/NaturalBoost, Formula and train-many are consumers
that test the abstraction boundaries. All application families remain required;
insurance, survival or any other family cannot stand in for the others. An
incumbent-compatible objective alone is not evidence of a distinctive advantage.

**Assessment:** the architecture has progressed from specifications to a working,
broad CPU foundation. Shared components now support structurally different
algorithms. We have not yet established cheaper independent authoring, competitive
real-data quality/cost, GPU execution, or repeated external use. The next investment
should test those hypotheses and close specific semantic gaps, rather than expand
the built-in objective catalog without new evidence.

| Product hypothesis | Evidence so far | Missing deciding evidence |
|---|---|---|
| Useful modifications require deeper access | D1–D5 specify objective, candidate, leaf, ordered-update and scheduling changes; incumbent hooks/source paths are documented | Fair implementation attempts showing where existing paths are actually costly |
| Shared components express different algorithms correctly | Public CPU operations, three growth policies, scalar/vector/residual leaves and twelve recipes have focused correctness tests | Installed public-only mutations and complete CPU conformance ledger |
| Agents can make those changes with less work | Mathematical verifiers and development cards exist | Frozen E5 attempts, including failures, appropriate controls and sealed held-outs |
| Workflow benefits outweigh runtime and adoption costs | Explicit preparation reuse, model persistence and installed examples work locally | Real selected comparisons, end-to-end cost, clean delivery matrix and independent repeated use |

## Progress and evidence boundary

F0.1 specification and F0.2 independent references are delivered. F0.3 evaluation
preparation remains incomplete. CPU construction has advanced through B03–B10
plus the A6/shared-preparation follow-ups; formal F1 exit has not passed. The
approved overlap is not a retroactive F0.3 pass. Close the outstanding preparation
ledger before interface freeze or formal comparisons.

The last full regression in [Sprint 037](037-shared-preparation.md) reports
**735 passing tests**: 247 public implementation tests plus 488 reference and
evaluation tests. This review independently collected the 247 public tests; it
did not rerun the full regression. Sprint 037 also records Ruff, strict MkDocs,
offline wheel/sdist and 19 installed-wheel examples passing on macOS,
Python 3.12.12 and NumPy 2.3.5. These results do not establish other platforms,
real-data quality, speed or adoption.

| Area | Implemented and exercised | Remaining boundary |
|---|---|---|
| Data and preparation | Owned numeric/mixed features, typed categories, weights, offsets, structure, class order, target/raw axes; explicit fitted preparation | Complete application provenance and evaluation binding; preparation reuse covers training binning/codes only |
| Tree foundation | Named additive fields, independent information channels, scoring/feasibility, routing; depthwise/best-first/symmetric growth; scalar/vector/routed residual leaves | Independent external authors must demonstrate replacement through installed public operations |
| Runtime | Immutable proposals and accepted/best models, explicit run identity, scoped RNG, rejection and failure isolation | Validation-driven patience/stop state; ordered update authoring; GPU residency/workspace/transfer semantics |
| Inference/artifacts | Validated raw/tree models, category/class metadata, AFT scale, frequency-severity roles and multi-output inverse scaling | Complete Normal/Formula dependencies, output units and evaluation metadata across workflows; broader clean-install matrix |
| Train-many | Sequential heterogeneous runs, shared prepared input, same-ID M=1/8/32 independent/reordered/regrouped equivalence and retained failures | Different validation stop rounds; real model selection; batching/fusion and measured full-set cost |
| Evaluation | Independent references, frozen data records, baseline workers, selection/integrity infrastructure and sealed held-out preparation | Current OpenBoost worker, full expected matrix, protocol enforcement and gate aggregation; formal author/quality/cost results |

All application rows below describe CPU mechanics, **not E3 acceptance**. Detailed
earlier findings remain in [Sprint 035](035-cpu-coverage-audit.md); Sprints 036–037
resolve its missing A6 recipe and repeated training-preparation findings.

| Required application | Current CPU path | Remaining application evidence |
|---|---|---|
| A1 regression | Squared-error recipe and numeric/mixed trees | Frozen real fit/selection/inference and quality comparison |
| A2 binary | Logistic geometry, explicit class order and probabilities | Real classification/calibration workflow and complete output metadata |
| A3 multiclass | Joint vector trees and multiclass bounds | Real multiclass comparison and class-schema workflow |
| A4 ranking | Query-local pairwise/lambda geometry and NDCG selection | Real source/query binding, declared pair limitations and ranking comparison |
| A5 quantiles | Routed weighted quantile and penalized leaf solvers | Temporal real evaluation, quantile reporting and installed leaf mutation |
| A6 multi-output | Independent/shared trees, train-fitted scaling and persisted inverse transform | Real grouped splits, scale-bound selection and per-target/final quality |
| A7 counts | Poisson with explicit exposure and rate/count transforms | Real frequency/deviance workflow and units/provenance |
| A8 positive targets | Weighted Gamma mean regression | Real eligibility, weights, positive-target comparison and units |
| A9 aggregate targets | Fixed-power Tweedie plus frequency-severity composition | Raw join audit, joint selection and aggregate quality |
| A10 survival | Event/right-censored log-normal AFT, fixed scale and survival inference | Source/license closure, censored NLL, IPCW/calibration and scale protocol |
| A11 distributional | Joint ordinary/Fisher Normal updates | Ordered-update mutation, distribution outputs, proper scores and calibration |
| A12 structured | Saturation Formula, explicit structure and full GGN | External formula dependency, real task plus recovery/misspecification evidence |
| A13 model selection | Shared preparation with independent heterogeneous runs | Independent stopping, frozen real ensemble selection and full-set cost |

**CUDA boosting is not implemented in the current v1 package.** RunContext
explicitly rejects non-CPU devices. Historical GPU work and baseline preflights
are useful evidence about their recorded revisions, not current v1 GPU results.

## Reflection and design risks

1. **Breadth has justified shared components; it has not proved authorability.**
   Recipe loops currently repeat policy wiring, and `_configuration`/`_trials`
   are private. An external author may still need substantial loop duplication.
   Test D2/D3/D4 early, record the actual obstacle, and expose only the smallest
   reusable operation that the evidence justifies. Do not preemptively build a
   generic trainer framework or count another built-in implementation as proof.
2. **Best-model selection does not implement early stopping.** The current
   recipes use fixed round budgets. M32 equality is useful, but does not verify
   independent validation-driven stopping. This is the next concrete correctness
   gap, not a performance optimization.
3. **Preparation reuse is narrower than end-to-end reuse.** Prediction still
   transforms inputs, and proposal/acceptance paths recompute ensemble predictions.
   These call paths are a possible cost/residency problem, not a measured speed
   regression. Profile representative shapes before redesigning caches. Preserve
   identity checks, rejection semantics and numerical equivalence if optimizing.
4. **Evaluation integration is now more valuable than more isolated smokes.**
   The baseline worker does not run current OpenBoost. The global matrix and
   independent gate aggregation remain incomplete. Reuse existing data/baseline
   artifacts, connect current recipes and expose missing cells explicitly.
5. **Adoption must remain an independent test.** Internal agents and repository
   authors cannot establish E7. An installable method package is a useful entry
   point; genuine demand requires another author to make and reuse a method.

## Execution plan and acceptance

The milestones are dependency groups, not single large commits. Open a bounded
sprint for each independently verifiable slice. M2 exploratory trials and M3
evaluation preparation should begin before all remaining CPU work is polished;
formal measurement still waits for its frozen prerequisites. M3 starts real-data
integration early, rather than postponing first contact until GPU completion.

### M1 — Independent stopping (next implementation sprint)

Mapping: R9/C4/A13, remaining F1.4 and D5 prerequisites.

- Define public stop policy/state separately from model acceptance. Track logical
  outer rounds, accepted-state version, trial attempts and stop reason distinctly.
  A line-search rejection must leave accepted/best model, raw values and RNG
  unchanged; individual backtracking trials must not consume patience.
- Observe the finite validation metric once per completed logical round, including
  a round whose proposals all reject. Keep the existing smaller-is-better metric
  contract. Specify positive patience, nonnegative finite min_delta, ties, initial
  baseline and zero-round behavior. Validation chooses stopping/best snapshots;
  training loss continues to govern the declared step-acceptance policy.
- Preserve strict best-model selection independently of the patience threshold.
  Record the last qualifying improvement for min_delta; ties do not reset patience.
  Return current and best models with explicit completed rounds and termination
  reason. Nonfinite metric failures remain visible and isolated to the affected run.
- Apply the same stop operation to built-in and external loops. Avoid encoding
  objective names in the generic runtime or scheduler.

**Acceptance:** a focused counterexample fails before implementation because
stop state is absent. Hand-calculated metric sequences verify threshold, tie,
rejection and boundary behavior. Shared-preparation M=1/8/32 runs use heterogeneous
K=1/2, different actual validation stop rounds, failure/retry and stable IDs;
independent, reordered and regrouped executions agree in stop state, predictions,
best snapshots and RNG. Existing fixed-budget behavior remains explicitly tested.
This establishes semantics, not fusion or speed.

### M2 — Public authoring and remaining CPU workflows

Mapping: C2–C7, R6/R7/R9, F1.5/F1.6 and exploratory F2.1/B11.

- First run D2 candidate-information and D3 penalized-leaf development tasks in
  separate installable extension packages against the wheel. They exercise deeper
  boundaries than another custom loss. Record public/private imports, core edits,
  duplicated logic, assistance and failures. These are exploratory, not scored E5
  or independent-adoption results.
- Exercise D4 joint-to-ordered Normal updates using public transactions; the next
  parameter reads the latest accepted state. Verify both orders, strict finite
  training-loss descent, at most six declared step sizes and complete rejection.
  Exercise changed update policy on Formula with its separate independent oracle.
- Complete D1 and D5 and all required public recipe/workflow entry points.
  Declare Normal/Formula inference dependencies and application output units;
  verify fresh-process persistence and prediction after training plugins are removed
  where core inference should be independent of those plugins.
- If an extension needs private imports or core edits, retain that failed attempt,
  repair the minimal boundary, then rerun the affected development checks. Do not
  compensate by putting each author task into a new built-in trainer.

**Acceptance:** all five E2 modification types execute through installed public
interfaces with independent math/state checks, zero required core edits/private
imports and no task-name branches in the generic runner. Two separate extension
packages and runnable A1–A13 CPU workflows produce reconstructible artifacts.
This is expressiveness evidence; comparative authoring cost remains M4.

### M3 — Close evaluation preparation and connect real applications

Mapping: B02/F0.3, C6/C7, B11 and early F4/E3 preparation.

- Reconcile the [F0 sequencing ledger](017-f0-sequencing-audit.md) against current
  artifacts. Close the A4 source/query path and A10 source-license gap; preserve
  existing hashes and corrected provenance rather than repeating resolved work.
- Freeze the full required recipe/application/device/fold/method matrix, output
  units, selection rules and environment/resource identities. Complete independent
  gate aggregation: absent, failed, contaminated or unsupported required cells
  cannot produce a passing gate. Test process budget and test-label boundaries,
  including the A6 train-scale/selection binding identified in Sprint 017.
- Add a current OpenBoost worker using the existing frozen data and result schemas.
  Start with a complete regression/Normal workflow and A6/A13 integration to expose
  output and selection issues, then connect every remaining application row. These
  are integration waves, not a reduction of required coverage.
- Each connected path must run train-only preparation, validation selection,
  sealed test evaluation, export and fresh-process prediction. Compare identical
  target units and splits; retain failures and all declared configurations.
- Run full quality searches as each task's protocol and recipe stabilize. Formal
  results require five declared splits/seeds, 16 configurations per method and all
  task-specific controls. Do not use four-round plumbing as quality evidence.

**Acceptance:** F0.3 has an explicit closing ledger; judges reject missing and
polluted evidence; all A1–A13 have runnable current OpenBoost/comparator bindings.
E3 then closes per application using committed raw results, at least six independent
sources and the existing thresholds. Do not mark all E3 passed because an early
integration wave passes. Cohort and CUDA-environment obligations also remain in
F0.3 and must be closed before the corresponding formal measurements.

### M4 — Freeze and test the central product hypothesis

Mapping: formal F1 exit, F2/B11, E0/E1/E2/E5.

- Audit CPU coverage and close E0/E1/E2 against actual public code, not reference
  totals. Complete F0.3 before freezing interfaces/judges and scoring comparisons.
- Freeze agent model/reasoning, tools/docs/cache, task/comparator versions, budgets
  and accounting. Smoke-test the attempt runner before running the full cohort.
  Allow each incumbent its strongest appropriate hooks, outer loop or source path;
  include Py-Boost where applicable. D1 is a control, not a presumed win.
- Run three attempts per task/arm, capped at 30 minutes or 20k generated tokens.
  Preserve unsuccessful attempts at the full cap in time aggregates. Separate
  exploratory runs, frozen development results and sealed H1/H2 evaluation.
  The foundation designer must not inspect held-out task/verifier contents; use
  the authorized independent evaluator. Tasks used in redesign lose held-out status.

**Acceptance:** E5 requires at least 12/15 correct development attempts and 2/3
per development task; at least 2/3 per held-out task and 4/6 overall; no core/private
dependencies; at least two deep-change types reduce capped median completion
time by 30% or more without fewer correct completions. Failure triggers a task,
documentation or abstraction review, not weaker opponents or hidden hints.

### M5 — Verified GPU execution and end-to-end cost

Mapping: F3/B12–B13, required CUDA R1/R4/R5/R6/R8 and R9, E1/E4.

- Profile CPU call paths and design explicit residency/transfer boundaries while
  earlier work proceeds. Implementation follows proven CPU semantics and F2
  interface revisions under the main phase order; exploratory profiling is not
  permission to declare F1/F2 complete.
- Build one complete resident path, then every required CUDA recipe subset,
  a nondefault author component and multiparameter adaptive updates. Verify
  gradients, statistics, chosen splits, leaves, every round, prediction and final
  metrics on real CUDA hardware. Report transfers, synchronization and fallback.
- After independent sequential parity, implement compatible shared execution and
  evaluate M=1/8/32. Separate preparation-only savings from batching/fusion. Use
  authorized Modal for bounded runs with frozen hardware and complete provenance.

**Acceptance:** existing E1 tolerances and E4 gates apply. Two quality-matched
medium/large standard workloads need warm-fit medians at most 2x the fastest
qualifying GPU control; public composition/optimized ratio at most 1.25. M=1
overhead is at most 10%; at least one of M=8/M=32 saves 20% full-set time and the
other is no more than 10% slower, with all required ensembles within that latter
limit. Include memory, startup, compilation, validation and inference costs;
standard CPU inference has its separate 2x gate. Historical P7 retains its original
threshold and is reported separately. A kernel benchmark cannot close this stage.

### M6 — Complete real value, delivery and adoption evidence

Mapping: F4/F5/B14, E3/E6/E7; continues the real-data work started in M3.

- Finish every A1–A13 result, including distribution/survival proper scores and
  calibration, Formula real/recovery/misspecification cases and real train-many
  selection. Each application passes independently; no cross-task average can hide
  a failure. Use the unchanged metric-specific E3 thresholds.
- Close CPU/CUDA clean-wheel and declared platform checks, all standard workflows,
  two extension packages, inference dependency removal and artifact reconstruction.
  Stabilize only contracts justified by these consumers and author experiments.
- Prepare a small install-to-baseline-to-mutation-to-verification package for
  external authors. Contact requires user authorization. E7 needs two independent
  authors, with one implementing their own method and reusing it on another task.
  This trial may start after F2 revisions, before all GPU work finishes.

**Acceptance:** engineering v1 means every required E0–E6 gate and individual
A1–A13 evidence pass. Adoption/impact remains unverified until E7 independently
passes. Finishing engineering does not justify an ecosystem or business-model claim.

## Immediate work and decision checkpoints

1. Implement and verify M1 stop semantics in the next bounded sprint.
2. Run the first installed D2/D3 development extensions; build the D4 public ordered
   update path around observed needs, not a speculative abstraction.
3. Begin M3's expected-matrix/current-worker integration alongside these CPU
   follow-ups. Deliver the first complete real-data workflow, then all required rows.

At every sprint closure, three implementation commits, phase transition or
correctness counterexample: record which hypothesis gained evidence, which gate
remains open and what can now be removed from the critical path. Repeated core edits
mean the foundation needs redesign. Real quality failures stay visible and must
be resolved without dropping required applications. Poor GPU end-to-end cost
requires profiling and an explicit failed gate, not a kernel-only success claim.

No additional built-in family should displace these deliverables unless it closes
an existing required contract or a concrete correctness/authoring counterexample.
No implementation, GPU run, outreach, push or publication is part of this review.

## Review verification

- Read public runtime/recipe/run call paths, sprint evidence and the required
  design/task/evaluation contracts. Preserved sealed held-out contents.
- Collected `tests/v1/test_public*.py`: 247 tests, not a fresh pass claim.
- Checked changed Markdown links, strict documentation build and diff whitespace.
  No production behavior changed; the last full regression remains Sprint 037.
- [Learning record](../learnings/2026-09-06-v1-goal-progress-review.md).
