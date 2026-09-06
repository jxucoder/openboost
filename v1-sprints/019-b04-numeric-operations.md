# Sprint 019: B04 numeric preparation and composable operations

Starting revision: f419b32. Status: initial operations slice complete. Approved B03–B06 overlap applies;
F0.3 remains open and every v1 application remains required.

## Plan and acceptance

1. Fit train-only numeric quantile cuts and produce owned feature-major int32
   codes with separate missing masks. Verify against the independent order-statistic
   oracle, including minimum cuts, duplicates, all-missing and unseen ranges.
2. Public row fields and exactly-once Newton weighting; independent information
   fields remain unweighted. Aggregate selected rows into per-feature histograms.
3. Public candidate enumeration, scoring, feasibility, choice, routing and scalar
   leaf solving. Compare histogram-prefix results with exhaustive original-row
   enumeration, including missing routes and custom cohort constraints.
4. Verify tests/docs and commit. Depthwise tree assembly, tree persistence and
   integration into runtime are the next B04 slice; this slice claims operations only.

First counterexamples: weights applied twice must fail; a high-gain split that
violates independent cohort mass must be replaceable through public feasibility.

## Result and reflection

Initial operations slice complete: 527 tests pass, including 18 public-operation
cases. Histogram-derived feasible candidates/gains/routing/leaves agree with the
independent original-row oracle. Both public documentation examples execute from
an isolated installed wheel; lint, strict docs and build pass. See the
[learning record](../learnings/2026-09-06-v1-b04-numeric-ops.md).

A coarser binning can erase a task's distinguishing candidate. Reference fixtures
must preserve the intended candidate set rather than silently turning an algorithm
change into a preprocessing comparison. Weight provenance and row/transformer
identities remain explicit through aggregation and partition.

Next B04 slice: assemble a depthwise tree through these public operations and
verify inference/persistence. Full B04, F1 and F0.3 remain incomplete. There is no
GPU or end-to-end boosting result from this slice.
