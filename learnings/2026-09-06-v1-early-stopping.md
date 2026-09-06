# 2026-09-06: Baseline validation stopping and selected-iteration replay

## Context

The search design declares patience 50, but baseline workers rejected stopping.
Native APIs differ in whether prediction defaults include trees after the best
iteration. A successful fit alone cannot verify saved-model selection semantics.

## Decision or Result

Require explicit validation targets and support nonunit validation weights and
A10 censoring. Use pinned native stopping metrics within trials, with independent
prediction-space metrics retained for cross-method selection. Save prediction
limits for XGBoost/NGBoost; preserve LightGBM's per-model best iteration and
CatBoost's truncated model. Count validation receives explicit exposure offsets.

## Changes

- [Worker](../benchmarks/v1/baseline_worker.py): explicit validation contract,
  native callbacks/validation pools, selected prediction limits and training JSON.
- [Stopping smoke](../benchmarks/v1/early_stopping_smoke.py) extends the weighted
  worker matrix and tests new-process reload on overfitting counterexamples.
- [Selection smoke](../benchmarks/v1/selection_smoke.py) accepts an optional
  stopping patience; this setting participates in its protocol identity.
- Input tests reject invalid patience, missing targets, invalid validation weights
  and unused validation fields when stopping is disabled.

## Verification

- `uv run --no-sync pytest tests/v1/test_baseline_worker.py -n 0 -q`: 15 passed.
- `OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -m benchmarks.v1.early_stopping_smoke`:
  30 CPU task/library cells passed; selected iterations match validation-history
  minima. XGBoost, LightGBM, CatBoost scalar and NGBoost distribution fixtures
  select round 1 before the final trained round and replay in new processes.
- [Raw stopping histories](../benchmarks/v1/evidence/early-stopping-cpu.json)
  identify code and the locked CPU environment. Patience three/max 24 rounds are
  smoke settings, not changes to preregistered real-search budgets.

## Failed Attempts

Inspection of the pinned source showed XGBoost and NGBoost retaining extra trees
by default. Explicit selected-round prediction avoids silently evaluating the
last trained ensemble. The earlier worker's rejection of stopping remains an
accurate historical boundary, superseded by this slice.

## Risks and Follow-ups

Native stopping metrics differ by method; do not describe this as a universal
custom-metric stopping implementation. Validation-selected independent task
metrics determine the opponent. GPU stopping remains unverified. Full real-task
matrix/worker isolation, ranking, composed/structured controls, auxiliary scores,
licenses and held-outs still prevent F0.3 exit.

## Commits

- This slice: `eval: support native baseline early stopping and replay`.

Final checks: 449 tests passed with no skips; Ruff and strict MkDocs passed.
The 16-trial selection smoke also passed with stopping enabled, including CLI
training metadata and selected-model new-process inference. Its
[summary](../benchmarks/v1/evidence/selection-early-stopping-cpu.json) records hashes;
full generated artifacts remain under ignored build output. Source hashes in both
new evidence files match the committed implementation bytes.

The unchanged fixed-round mode also passed all 30 CPU cells using
`OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 build/v1-env/bin/python -c 'from benchmarks.v1.worker_smoke import run; print(len(run()))'`;
this rerun did not overwrite the historical fixed-round artifact.
