# 2026-09-06: Compose A5 quantiles through public scalar recipes

## Context

Parent f9bf01f. The frozen Bike exporter existed, but the current OpenBoost worker
only accepted A1/A2/A3/A6/A11. Historical baseline integration was not evidence
that the redesigned foundation supported the real A5 workflow.

## Decision or Result

Compose three independent public quantile recipes at frozen levels 0.1/0.5/0.9.
Persist explicit ordered models and level metadata, separate stopping and best
selection, and raw predictions. Report crossings without post-hoc sorting.
No new public/core primitive was necessary.

## Changes

- Current worker, inference bundle and smoke harness now support A5.
- Tests cover weighted direct parity, stopping selection, fresh replay, malformed
  schemas and preservation of crossing predictions.
- [Sprint 053](../v1-sprints/053-quantile-worker.md) records scope and reflection.

## Verification

- Six initial tests failed on unsupported A5 before implementation.
- All five frozen Bike origins pass with exact fresh inference and source IDs.
- Independent NumPy pinball recomputation matches each selected score.
- Source/output hashes match [raw evidence](../benchmarks/v1/evidence/quantile-053/README.md).
- `uv run --no-sync pytest tests/ -m "not gpu and not benchmark" -q --tb=short`: 858 passed.
- Ruff, strict MkDocs and diff whitespace passed; macOS/Python 3.12.12/NumPy 2.3.5.
  Commands use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

## Failed Attempts

The missing adapter was reproduced without changing the frozen data split or
objective. No quality failure was corrected or hidden by this integration.

## Risks and Follow-ups

Independent levels may cross. No observed crossings here is not a guarantee.
Real searches, calibrated quality, remaining adapters, D5 and CUDA stay open.
Next address A7 count/exposure binding; offsets must not be silently ignored.

## Commits

- This current A5 worker slice; parent f9bf01f. Local only, no push.
