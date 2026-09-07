# 2026-09-06: Bind count exposure exactly once in current evaluation

## Context

Parent 003fdac. The frozen A7 worker packet contains period counts and separate
exposure vectors. The public Poisson recipe already represents exposure as a
structure role, but the current evaluation worker had no A7 path.

## Decision or Result

Bind exposure to the public structure role without adding it to weights or
supplying a second log offset. Persist raw log-rate models with an explicit
period-count output tag. Inference requires new positive aligned exposure.
Extra offsets and exposure on foreign applications are rejected.

## Changes

- Current worker and inference support explicit A7 count/exposure semantics.
- Real-data harness includes exposure in prediction-only replay packets.
- Tests cover weighted final/best direct parity, independent likelihood,
  exposure scaling, fresh persistence and malformed/foreign inputs.
- [Sprint 054](../v1-sprints/054-count-worker.md).

## Verification

- Both new direct-parity cases failed on missing A7 support before implementation.
- Current worker suite: 47 passed. Full CPU suite: 866 passed.
- Commands use UV_CACHE_DIR=/tmp/openboost-research-uv-cache and uv run --no-sync.
- All five full-data folds pass with exact fresh exposure-aware replay.
- Independent NLL checks pass at rtol=1e-12/atol=1e-14; the initial 1e-13 relative
  check failed at a 2.4e-14 absolute discrepancy. Raw differences and math.fsum
  checks are retained in [evidence](../benchmarks/v1/evidence/count-054/README.md).
- Ruff, strict MkDocs and diff whitespace checks passed.

## Failed Attempts

Missing adapter support was reproduced without altering the mathematical recipe.
No exposure-as-weight approximation or implicit unit-exposure fallback was used.

## Risks and Follow-ups

Real-data integration is not a quality search or calibrated rate/deviance result.
All remaining adapters, searches, D5, CUDA and adoption gates stay open.

## Commits

- This A7 worker slice; parent 003fdac. Local only, no push.
