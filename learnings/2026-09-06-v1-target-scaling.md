# 2026-09-06: Preserve A6 target units through comparator training

## Context

The task card requires train-only target standardization and a saved inverse.
The numeric worker previously fit raw targets, so large-unit outputs dominated
shared multi-output fitting and native validation stopping.

## Decision or Result

Fit unweighted per-column training mean/std; constant columns use std one.
Transform validation with those same statistics and use zero standardized bases.
Fit and replay return original target units. Explicit training weights remain
training weights and do not redefine the preregistered normalization.

## Changes

- Worker saves target scale in the model bundle and training receipt.
- CPU smoke includes targets with very different units and a constant column,
  verifies unchanged caller arrays and new-process A6 reload.
- Invalid scalar and empty-column A6 targets fail before library import.

## Verification

- `build/v1-env/bin/python` invoking `worker_smoke.run()` and `run(3)`:
  30 fixed and 30 stopping CPU cells passed. Raw metadata/results:
  [worker-target-scale-cpu.json](../benchmarks/v1/evidence/worker-target-scale-cpu.json).
  A6 fresh-process replay passed for all three comparators.
- `uv run --no-sync pytest tests/ -n 0 -q`: 474 passed.
- Ruff across production/evaluation/tests and strict MkDocs passed.

## Failed Attempts

The expanded fixture exposed CatBoost rejecting a constant target column. The
A6 adapter explicitly enables allow_const_label while retaining MultiRMSE and
zero initialization. This is needed for the task's constant-target contract.

## Risks and Follow-ups

CPU synthetic checks do not establish real-data quality or CUDA target-space
parity. Prior raw artifacts retain their original code identity; do not reinterpret
them as target-standardized runs.

## Commits

- This slice: `eval: preserve train-only multi-output target scaling`.
