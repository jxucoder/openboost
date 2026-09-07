# 2026-09-06: Keep Formula structure outside parameter-tree inputs

## Context

Parent 3ff97c3. The A12 exporter provided ordinary GBDT inputs with appended age,
but the current Formula recipe requires age as a separate structural argument.

## Decision or Result

Emit a separate formula-input packet retaining the identical encoded composition
features and normalized age/28. Bind two raw parameters and public saturation
prediction, with explicit structured inference. Preserve the ordinary packet.

## Changes

- Exporter/current worker/inference/smoke add structured A12 integration.
- Tests cover weighted recipe parity, fresh replay, missing/invalid inference age
  and exact exclusion of appended age from tree predictors.
- [Sprint 060](../v1-sprints/060-structured-worker.md).

## Verification

- Full CPU regression: 912 passed. Ruff, strict MkDocs and whitespace pass.
- Five frozen folds pass with exact replay; independent formula/score and exact
  feature/age/source-ID checks pass. Source/output hashes match
  [evidence](../benchmarks/v1/evidence/structured-060/README.md).
- Commands use uv run --no-sync and UV_CACHE_DIR=/tmp/openboost-research-uv-cache;
  real export additionally uses --offline --with xlrd. macOS/Python 3.12.12/NumPy 2.3.5.

## Failed Attempts

The initial binding test ignored preprocessing missing-indicator columns; changed
it to assert removal of exactly the appended age column. Initial real export
failed due to missing xlrd, before fitting. Cached ephemeral dependency retry
passed; no project environment mutation or source change.

## Risks and Follow-ups

No real extrapolation/global-control/quality acceptance. A4, searches/D5 and CPU
phase gates remain; GPU is unimplemented. The adapter cannot prove the units of
arbitrary standalone packets; frozen exporter hashes bind those semantics.

## Commits

- This structured A12 slice; parent 3ff97c3. Local only, no push.
