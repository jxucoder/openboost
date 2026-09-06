# Sprint 016: Complete evaluation preparation

Starting revision: `dc74401`. Status: in progress. Mapping: B02/F0.3.

## Plan and acceptance

1. Freeze remaining real data and task-specific five-split semantics, source/license
   evidence, preprocessing, and source/array/row hashes for every A1–A13 application.
2. Verify installed comparator capabilities on CPU then real CUDA; freeze versions,
   environments, 16 configurations per method, and resource limits before quality runs.
3. Implement process execution and independent quality/selection judging with
   adversarial checks; freeze author evaluation inputs without contaminating held-outs.
4. Audit every F0.3 exit requirement, record unresolved evidence explicitly, and
   commit independently verified slices. Do not mark F0 complete based on test totals.

First counterexamples: rows from one subject/recipe must never cross data partitions;
invalid or missing comparator output must not yield a passing quality gate.

## Reflection and evidence

Pending. Public production components remain F1 work. External source failures,
package availability, and held-out independence must be resolved without silently
changing the required application scope or evaluation thresholds.

### Data slice

Covertype, Parkinsons, Concrete, Veteran, and insurance inputs/splits are frozen.
Nine focused tests passed, including source corruption and hand-worked joins.
Every actual source was parsed and replayed; no model quality was inspected.
The join excludes 195 orphan claims and identifies 9,116 positive-count/no-payment
policies. Grouped partitions preserve subjects, recipes, and policies. This
validates target/data semantics without creating any new production implementation.
Remaining source blockers: MSLR retrieval/agreement and Housing/Veteran license
confirmation. Required applications and all original gates remain intact.

### Evaluation machinery slice

Actual train-only encoding hashes now cover available data, with separate A8/A9
training populations and A10 censoring support. Independent metrics and five-fold
comparisons retain every target/quantile; altered prediction row order fails even
when file hashes are recomputed. Worker zero exit without artifacts, nonzero exit,
and timeout fail. Seventeen focused checks and 422 total tests passed.

Reflection: a real CUDA comparator abort showed why the evaluation process boundary
matters. Artifact/metric checks and baseline probes remain distinct from complete
task execution. Full validation-selection receipts, expected recipe/device matrix,
auxiliary task metrics and held-out cohort integration are still unfinished;
partial modules must not create an F0.3 completion label.

### Comparator capability slice

A real T4 isolated-process matrix completed: 29 CPU / 28 CUDA passing built-in
cells, four/five explicit unsupported cells. Nonunit weights, native mixed data,
external offset persistence and secondary algorithms have separate CPU artifacts.
See the [evidence index](../benchmarks/v1/evidence/README.md).

Reflection: the standard LightGBM wheel is not the CUDA comparator. Source builds
and process isolation were necessary. Keep the failed build, native abort and
original tolerance mismatch; a later passing preflight does not erase failures
or prove mixed-library context safety. No end-to-end quality/speed claim follows.

Py-Boost 0.5.2 also passed weighted scalar/vector GPU fitting and JSON reload.
Its initial zero-verbosity callback failure remains in the evidence directory.
These checks do not cover all Py-Boost E5 arms or real task metrics.

### Validation worker slice

The fixed-round numeric worker passed 30 CPU task/library fit/reload cells and
exposure doubling in the three count adapters. Nine adversarial checks reject
invalid exposure, censoring, weights/IDs and unsupported options/test arrays.
The complete suite is 431 passing tests. This worker explicitly rejects the
search design's early-stopping option; search execution is not ready to freeze.

### Current exit audit

F0.3 remains **in progress**, and F1 has not started. Remaining acceptance work:

1. Resolve A4 ranking source/agreement and Housing/Veteran license evidence.
2. Complete query-aware, composed A9, structural A12 and A13 adapters; implement
   early stopping and the validation-selection/test-release path, with corruption
   and leakage counterexamples. Bind the independently expected full matrix.
3. Complete auxiliary application reports, including classification and IPCW
   survival metrics, plus paired uncertainty reports and the E-gate reconstruction.
4. Freeze exact native builds, agent cohort/settings/accounting and E4 workloads.
5. Freeze H1/H2 outside the foundation designer's context. A separate evaluator
   requires the user's delegation authorization; the pending request is unanswered.
6. Re-run the complete F0.3 audit and close this sprint only with all requirements
   satisfied. Neither comparator smoke totals nor reference tests waive a gap.

### Next slice: validation selection and test release

1. Add adversarial fixtures for missing/failed trials, altered configurations,
   invented validation scores, changed model bytes and overlapping row IDs.
2. Independently recompute every trial's validation metrics against a separately
   pinned protocol; select across all declared methods and seal an immutable receipt.
3. Require the pinned receipt and unchanged artifacts before reading test features.
   Verify the boundary with a synthetic complete 16-trial search and document the
   distinction between this API ordering and operating-system access isolation.

Selection slice result: 12 adversarial tests passed, and a complete synthetic
16-trial XGBoost subprocess search passed audit/seal/release followed by new-process
selected-model inference. Scores are recomputed independently; the receipt binds
all trial outputs. Test features are not read by the selector. A forged winner,
changed model, omitted trial or row overlap fails.

Reflection: content hashes alone do not prove that a worker never accessed test
files. The API enforces sequence and consistency; restricted mounts/execution
provenance still need integration. The full required real-data matrix, early
stopping, auxiliary metrics and independent held-outs remain open. Do not turn
this synthetic orchestration success into a formal E3 or F0.3 pass.

### Next slice: baseline early stopping

1. Require explicit validation targets/weights (and censoring for A10) when early
   stopping is enabled; reject malformed or unused validation fields.
2. Use pinned native stopping implementations and retain the selected tree count
   in saved prediction state. Record each method's stopping metric and history;
   independent cross-method selection continues to use prediction-space metrics.
3. Exercise scalar, vector, quantile, offset and distribution paths in the locked
   CPU environment, with a counterexample whose best iteration precedes the last.
   Verify replay of the selected iteration, including a new-process check.

Early-stopping result: the 30 supported CPU task/library cells passed weighted
validation stopping and replay. Selected rounds agree with recorded native metric
minima. Deliberate overfitting cases select round 1 and reproduce after loading in
fresh processes across XGBoost, LightGBM, CatBoost and NGBoost. Patience-three smoke
settings do not change the preregistered patience of 50.

Reflection: native default prediction is not universally best-iteration prediction.
Preserve limits explicitly for XGBoost/NGBoost and retain each LightGBM fit's limit.
Record native stopping metrics separately from independent method selection.
CPU evidence does not establish CUDA stopping or complete F0.3 acceptance.

### Remaining adapter work

Implement A4 query-aware workers first: contiguous groups, disjoint query IDs,
explicit per-query weights, native ranking objectives/stopping and saved scores.
Reject row weights and fragmented groups before entering a comparator. Then add
parametric/composed A9/A12 controls and auxiliary metrics against hand-worked
oracles. Source/license and held-out requirements remain separate exit conditions.

Ranking adapter result: all three CPU rankers passed weighted query-aware fitting,
named-NDCG stopping and reload. Five group/weight/overlap counterexamples passed.
The original smoke's metric-order assumption failed on CatBoost and is retained;
selecting a metric by position is unsafe when a library adds a second history.
Real ranking data/agreement remains unresolved; A4 quality is not passed.

Parametric adapter result: A7/A8/A9 GLMs, matched paid-count × severity, and the
A12 global formula now execute through strict validation workers. Five subprocess
checks matched independent hand-calculated outputs. Six input/math tests reject
invalid joins/structure. The full suite is 460 passing tests. Composition penalty
pairs are declared before real quality evaluation.

Reflection: paying claims and recorded claims are distinct target populations.
Validate exact paid-record reconstruction before fitting either component; the
same output units must survive model composition and reload. These controls do
not implement FormulaBoost's coupled updates or finish full task/matrix binding.

Auxiliary slice: weighted classification diagnostics, Normal PIT, supported-grid
IPCW Brier, separately named Harrell C, training-age support errors and descriptive
paired intervals are implemented. The quality report retains complete fold metrics
and explicitly lists missing A10/A12 auxiliary inputs. Rehashed invalid support
still fails. Thirteen focused metric/artifact tests pass.

Reflection after these three implementation slices: component-level adapters and
judges now cover more task semantics, but a complete pinned real-data execution
matrix, outer coupled controls, A13 scheduling, native-build/agent provenance and
held-out independence still require work. Green synthetic checks cannot close F0.3.

### A6 target-space correction

Plan: standardize all comparator A6 targets from training rows only, preserve the
constant-column rule and zero standardized base, invert predictions after fit and
reload, then exercise mismatched target scales and fresh-process replay. Native
stopping metrics operate in standardized target space; final metrics use original
units and the frozen training scale. This corrects a task-contract gap before
real-data trials.

Result: 30 fixed and 30 stopping CPU cells pass, including fresh-process A6
replay across three comparators. All 474 tests, Ruff and strict MkDocs pass.

Source review: the exact Housing archive matches Figshare version 2's MD5 and
CC BY 4.0 declaration. A dated overlay preserves the original freeze and records
uploader attribution. MSLR agreement retrieval still returned HTTP 401; Veteran
original-source license remains unresolved.

Held-out preparation: user authorized a separate evaluation agent, which sealed
H1/H2 cards/verifiers and reported passing internal/adversarial validation. Only
hashes and status reached the foundation designer. The opaque package is retained
in ignored local storage and its hash is in the public held-out manifest. This
is separate-context authorship, not OS isolation or a completed E5 cohort.

Reflection: task-space normalization is part of comparator fairness, not cosmetic
preprocessing. The constant-target counterexample found a real native-library
contract difference. Held-out custody now exists, but does not remove the need
for restricted candidate execution and frozen cohort accounting.
