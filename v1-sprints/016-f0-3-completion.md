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
