# Sprint 021: B05 tree terms and squared boosting

Starting revision: 85e9333. Status: complete for this slice. Approved B03–B06 overlap applies;
F0.3 remains open and all required use cases remain in scope.

## Plan and acceptance

1. Replace the constant-only model with a versioned ensemble of constant and
   mapped scalar-tree terms; retain immutable transaction and best-state semantics.
2. Compose a complete scalar squared recipe with weighted derivatives, offsets,
   fixed steps and bounded backtracking that reuses the learner. Compare multiple
   rounds and intermediate values with the independent exhaustive reference.
3. Verify rejection/foreign proposals, mixed-term persistence, offsets, best
   snapshots and fresh-process inference. Run regression/lint/docs/build, reflect,
   and commit locally. Normal geometry/recipes remain the next B05 slice.

Smallest initial failure: importing the complete squared recipe for the two-round
reference comparison. No new benchmark quality or performance claim is intended.

## Result and reflection

The first complete CPU squared recipe is implemented with mapped tree terms,
fixed/backtracking steps and inference persistence. All 560 CPU tests pass;
independent three-round intermediates agree, and rejection/offset/best-state
checks pass. Ruff, strict docs, build and all four installed-wheel examples pass.
See the [learning record](../learnings/2026-09-06-v1-b05-squared-recipe.md).

Observation: scalar boosting now composes the same public operations used by
manual algorithm changes. Atomic term tuples allow multiple mapped updates
without specializing runtime for squared loss. Evidence still only validates
scalar-target geometry: Normal will require separating target width from raw
parameter width, currently coupled in Problem/state. Address that next rather
than duplicating labels or building a separate distributional trainer. The CPU
implementation recomputes predictions during trials; optimize after the geometry
and Formula/heterogeneous probes establish the right boundary.

Normal remains the next B05 slice; B06 must follow before stabilization. F0.3 and
all required evaluation gates remain open. No scope was dropped and nothing pushed.
