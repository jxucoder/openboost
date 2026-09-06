# Sprint 050: Bounded full-data CPU profile

Parent: 078ca61 (merged PR 23). Status: profiling slice complete; full A3 validation still incomplete.

## Plan and acceptance

1. Add a diagnostic wrapper that retains cProfile statistics and periodic Python
   stacks when a worker reaches a soft deadline. Keep an outer process kill cap.
2. Test successful and interrupted callbacks; profile the unchanged Sprint 049
   fold-zero packet/configuration on the full input with a 60-second soft deadline
   and 90-second hard limit, one thread. No core/recipe edits or budget increase.
3. Identify measured call paths, record overhead/limitations and exact artifacts,
   update next implementation step and commit locally without pushing.

An interrupted profile is not a passing fit or comparable timing benchmark.

## Results and reflection

The diagnostic ended intentionally at its 60-second soft deadline (exit 124);
the outer process reports error, not a successful fit. Raw cProfile statistics
and 20-second stacks were retained. In this interrupted instrumented window:

- histogram: 20 calls, 29.48 seconds self / 29.97 cumulative;
- hashlib update: 11,228 calls, 22.32 seconds self;
- candidates: six calls, 23.55 seconds cumulative;
- depthwise: two calls, 54.54 seconds cumulative;
- binning transform: seven calls, 2.28 seconds cumulative.

Cumulative times overlap and must not be added. The profile is incomplete and
instrumented; it is not an end-to-end speed result. Local regression tests also
ran during the beginning of the diagnostic window. Stacks at 20/40 seconds show
histogram and candidate identity hashing respectively. One round reached update
transactions before the deadline; a second tree-growth call was active.

Inspection confirms candidates recomputes `_identity(hist.rows)` for every
candidate even though the immutable row set is constant for that invocation.
This falsifies the earlier priority hypothesis that prediction-time binning was
the main cause in this window. Next hoist this invariant hash, preserving exact
candidate identities and statistics, then measure the same bounded workload.
Histogram optimization remains a separate measured opportunity, not a speculative
rewrite bundled into the first change.

Two diagnostic tests verify success/deadline artifacts and timer restoration.
Full CPU regression: 846 passed; Ruff and strict docs pass. Source/raw artifact
hashes match [profile-050](../benchmarks/v1/evidence/profile-050/README.md).
No core code changed and A3 remains incomplete. See
[learning](../learnings/2026-09-06-v1-covertype-profile.md).
