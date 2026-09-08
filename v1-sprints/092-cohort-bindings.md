# Sprint 092-C/D: Bind revised cases and freeze hardware validation

Status: locally approved construction after `2c5927c`. No upload or eighth device
invocation is authorized. The concrete run request is the final deliverable.

## Execution plan

1. Add a separate original-row Normal trajectory oracle with objective comparison
   for training and validation best. Preserve the original oracle and its hashes.
   Validate the new oracle on the two recorded failures, tiny/equal-score changes,
   all ninety historical settings and their differences from the old trajectories.
2. Bind all 383 historical requirements to collected cases. Keep 236 unchanged
   operation checks; create explicitly named revised transaction/recipe/D2 cohorts
   with all original settings and tolerances. Capture and independently judge
   comparisons at actual stored inputs as well as checking complete trajectories.
   Include the additional 117 operation and 27 consumer distinctions. Preserve the
   original mapping in a separate immutable file; add a new binding record.
3. Prepare an exact-source run-8 protocol, bounded artifacts, separate historical
   and revised verdicts, installed D2/fresh CPU inference, numerical lowering
   evidence and end-to-end cost accounting. Test failure-closed dispatch locally
   and collect from the exact upload snapshot. Commit the freeze and request one
   concrete upload/run allowance only after it is reviewable.

## Numerical and accounting rules

The existing float64 original-row grower remains the structurally independent
geometry/tree/prediction oracle. The new reference loop uses the frozen 092-A
interval comparison and keeps a separate best validation snapshot. This is CPU
mathematics, never emulated device evidence. Preserve all original metric/raw/leaf
tolerances and strict well-separated split/trial decisions. Record any changed
reference trajectory before dispatch. Do not alter test settings to seek a pass.

Comparison instrumentation also reads the actual prepared target/offset/weight
and the exact before/after device buffers. Check bounds against the independent
original-row high-precision expression. It runs after the actual comparison and
cannot supply an algorithm decision. Equal/unresolved cases retain their reason;
neither rounded total-loss subtraction nor a blanket epsilon is a replacement.
Instrumentation adds exports/synchronizations, so its timing is diagnostic only.
Measure end-to-end fit separately without comparison instrumentation.

Historical tests retain their original source and outcomes. Their current-code
run is reported separately; the two known full-loss failures and the new best
snapshot's effect on old recipe memory assertions are explicit expected
disagreements, not repaired historical tests. A revised gate requires its own
complete pass and artifacts. Missing, skipped, duplicate or unexpected cases fail.

Acceptance for this local task is complete bindings, honest recorded CPU checks,
isolated collection and a concrete frozen request. Actual CUDA acceptance remains
the subsequent separately authorized run and retrospective.

## Binding construction result

The [collected bindings](092-collected-case-bindings.json) cover all 383 original
requirements with 236 unchanged and 147 explicitly revised cases. All original
sources and tolerances remain intact. The [reference artifact](092-reference-trajectories.json)
retains both complete summaries for all ninety settings: none changes on the
independent float64 reference. Tiny/equal-score distinguishing tests separately
establish that the new reference accepts proved improvements and advances best.

Each revised CUDA test installs a read-only comparison audit that runs the actual
operation first, preserves its result, records exact input bits and checks its
bounds independently. Instrumented timing is labelled explicitly. The nineteen
revised saved-model replays use a separate artifact root from historical outputs.
All 147 revised cases collect; none has run on CUDA. Hardware protocol construction
is the remaining local step.
