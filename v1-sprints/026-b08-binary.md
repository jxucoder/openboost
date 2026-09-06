# Sprint 026: B08 class schemas and binary logistic recipe

Starting revision: 082b614. Status: binary slice complete.

## Plan and acceptance

1. Fit immutable typed class schemas on training labels and bind encoded labels
   explicitly to Problem/model identity; reject unknown validation labels.
2. Compose numerically stable binary logistic geometry with shared tree operations
   and transactions, including weights, offsets and fixed/backtracking steps.
3. Persist class order with raw ensembles; verify probabilities/decoded labels,
   independent multi-round geometry, fresh-process inference and corrupt schemas.
4. Run regressions/lint/docs/build, reflect and commit. Multiclass/vector leaves
   remain the next B08 slice; no full classification parity or CUDA claim.

First failing check: importing the class schema and fitting typed training labels.

## Result and reflection

Binary logistic boosting, class-bound state and probability/label persistence are
implemented. All 621 CPU tests pass. Independent three-round geometry/tree checks,
extreme logits, offsets, schema errors and fresh-process inference pass. Lint/docs/
build and every example across eight installed-wheel pages pass. See the
[learning record](../learnings/2026-09-06-v1-b08-binary.md).

Observation: binary loss reused scalar split/leaf operations and transactions,
but label meaning had to become explicit state rather than recipe-local metadata.
The schema now follows best models and serialized probability columns. Multiclass
will next test shared vector topology and separate per-channel bounds; do not
approximate that requirement by declaring several scalar models complete coverage.

Multiclass and vector leaves remain next in B08. F1, F0.3 and full application/
quality/agent/CUDA evaluations remain incomplete. No interfaces frozen, no parity
claim and nothing pushed.
