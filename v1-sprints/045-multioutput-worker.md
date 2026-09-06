# Sprint 045: A6 scaling-bound evaluation worker

Parent: 374ab87. Status: complete for A6 validation integration. Mapping: Sprint 038 M3, A6.

## Plan and acceptance

1. Reproduce unsupported A6, then connect shared/independent multi-output recipes.
2. Fit the frozen unweighted target-scale convention on training targets only;
   use identical scaling for validation selection and persist inverse output units.
3. Test heterogeneous/constant targets, sample weights, shifted validation targets,
   corrupt scale metadata and fresh-process predictions for both growth modes.
4. Run five frozen Parkinsons folds, compare scale metadata to the freeze, run
   regression/lint/docs, record reflection and commit locally.

This is four-round validation integration, not A6 quality acceptance or A13 search.
No test truth is scored. Existing A1/A11 behavior must remain unchanged.

## Results and reflection

A6 now composes current multi_squared (shared or independent trees) with public
TargetScale/MultiOutputModel. The benchmark scale is computed from training
population means/stds using the existing frozen preprocessing operation, then
applied to train and validation. We do not substitute public weighted scale fitting:
the evaluation protocol uses unweighted scales, while sample weights still affect
training/validation losses. This distinction is explicit in saved metadata.

Original-unit output is preserved in a JSON evaluation bundle carrying validated
scale metadata. Twenty worker tests pass (five new), including both modes with
and without patience, widely different target units, constant channels, zero and
nonuniform weights, shifted validation targets and corrupt scale rejection.
Fresh-process predictions exactly match direct public recipe/wrapper execution.
All five real grouped Parkinsons folds pass and scale metadata exactly matches
the committed freeze. These runs use shared mode; independent mode is tested on
synthetic fixtures, not claimed as a five-fold real result.

Full CPU regression: 819 passed. Ruff and strict MkDocs pass. Raw evidence:
[multi-worker-045](../benchmarks/v1/evidence/multi-worker-045/README.md).
See [learning](../learnings/2026-09-06-v1-multioutput-worker.md).

The protocol mismatch was resolved at the adapter boundary without changing core
scaling semantics or weakening the freeze. This is validation integration, not
full A6 acceptance: four rounds/32 bins, no configuration search, test score,
comparison or OS label isolation. A13 selection, remaining application adapters,
D5 probes and F0.3 closure remain next. No GPU or performance claim.
