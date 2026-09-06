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
