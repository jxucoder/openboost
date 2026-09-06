# 2026-09-06: Bind frozen real folds to comparator inputs

## Context

Synthetic comparator checks did not verify actual dataset-to-worker wiring.
A6 normalization and A12 structure inputs made implicit packet assembly unsafe.

## Decision or Result

Export Housing A1/A11, Parkinsons A6 and Concrete A12 ordinary-GBDT inputs only
after all five folds agree with the frozen source/encoder/row identities. Keep
worker validation arrays separate from test features and test truth. This is
an evaluation-side preparation operation, not an OS sandbox.

## Changes

- `benchmarks/v1/worker_data.py`: verified source binding, original-unit targets,
  disjoint groups, A12 age feature and separate structural support.
- `benchmarks/v1/worker_data_smoke.py`: bounded fresh-process real validation fits.
- Seven focused counterexamples cover row order, changed encoder/target, duplicate
  rows, crossed groups, output-unit preservation and structure alignment.

## Verification

- All 481 v1 tests passed, including seven new packet counterexamples.
- Ruff across production/evaluation/tests and strict MkDocs passed.
- Twenty real-data validation fits (four application paths, five folds) passed
  in fresh CPU processes. XGBoost covers A1/A6/A12; CatBoost covers A11.
  [Raw summary](../benchmarks/v1/evidence/real-worker-binding-cpu.json) records
  exact commands, source/data/packet hashes, versions, stopping and artifact hashes.
  Replay is checked inside each worker before success. No test truth was scored.

## Failed Attempts

Initial lint requested combining nested context managers in the smoke. No
mathematical behavior or frozen inputs changed to resolve it.

## Risks and Follow-ups

This exporter does not cover A2/A3/A4/A5/A7/A8/A9/A10/A13 or coupled controls yet.
Four-round validation plumbing cannot establish predictive quality, complete
16-trial selection, GPU execution or test-access isolation. Held-outs remain sealed.

## Commits

- This slice: `eval: bind frozen real folds to comparator worker packets`.
