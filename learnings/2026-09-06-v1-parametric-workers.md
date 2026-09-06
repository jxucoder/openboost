# 2026-09-06: Parametric and paid-loss comparison workers

## Context

A9 requires a matching paid-frequency/severity baseline, not raw ClaimNb multiplied
by positive-only severity. A12 requires the same global formula as a control.
Earlier mathematical probes did not provide runnable validation worker adapters.

## Decision or Result

Add positive-family GLMs, a paid-record-validated composition and positive global
formula fit. Business/exposure weights enter once with explicit output units.
Training-fitted scaling is saved. Invalid joins and unsuccessful convergence fail.

## Changes

- [Controls](../benchmarks/v1/parametric.py) and strict
  [worker](../benchmarks/v1/parametric_worker.py).
- [Hand-check smoke](../benchmarks/v1/parametric_smoke.py) and
  [subprocess smoke](../benchmarks/v1/parametric_worker_smoke.py).
- Six mathematical/input counterexamples and 16 preregistered composition penalty
  pairs; no real quality data was used to choose their range.

## Verification

- `uv run --no-sync pytest tests/v1/test_parametric_controls.py -n 0 -q`: six passed.
- Pinned CPU `python -m benchmarks.v1.parametric_smoke`: five controls passed
  independent constant-feature weighted-mean formulas, exposure doubling and
  persistence; the nonlinear formula recovered known parameters.
- Pinned CPU `python -m benchmarks.v1.parametric_worker_smoke build/v1-parametric-workers-001`:
  all five workers passed bounded CLI execution and prediction/row-ID checks.
- [Raw control evidence](../benchmarks/v1/evidence/parametric-cpu.json) and
  [worker evidence](../benchmarks/v1/evidence/parametric-worker-cpu.json).
- Full suite: 460 passed, no skips. Ruff and strict MkDocs passed.

## Failed Attempts

No failed numerical fit in this slice. Explicit hand-worked count/total mismatches
and orphan/nonpositive claims fail before either composition model fits.

## Risks and Follow-ups

These are CPU benchmark controls, not OpenBoost production models. Pickle bundles
are for trusted local artifacts only. Real task/source/search binding, outer coupled
controls, auxiliary scores and held-out work remain open. Global formula recovery
on an identifiable synthetic curve is not real-data parameter identification.

## Commits

- This slice: `eval: add parametric and paid-loss comparison workers`.
