# 2026-09-06: Bind event/right censoring in current A10 evaluation

## Context

Parent 20d9b7c. The public fixed-scale AFT recipe existed, while the current
real-data worker did not accept the frozen time/event packets.

## Decision or Result

Convert exact events to equal positive bounds and right censoring to positive
lower/infinite upper bounds. Fix sigma=1 explicitly and persist its metadata.
Emit [log-time location, sigma], matching the evaluation schema. Validation
selection uses censored likelihood, not squared error or treating censoring as death.

## Changes

- Current worker and inference add A10 with strict time/event/scale validation.
- Weighted final/best direct parity and fresh replay tests.
- Independent erfc-based censored NLL checks and malformed input rejection.
- [Sprint 059](../v1-sprints/059-survival-worker.md).

## Verification

- Both direct A10 tests failed on unsupported application before implementation.
- Current worker suite: 70 passed.
- Full CPU regression: 909 passed. All five real folds pass exact replay and
  independent censored NLL checks; source/output hashes match
  [evidence](../benchmarks/v1/evidence/survival-059/README.md).
- Ruff, strict MkDocs and whitespace pass. Commands use uv run --no-sync with
  UV_CACHE_DIR=/tmp/openboost-research-uv-cache; macOS/Python 3.12.12/NumPy 2.3.5.

## Failed Attempts

A target_kind keyword was initially placed before a positional argument; lint
and test collection caught the syntax error. Corrected before running workloads.
No mathematical support, source split or evaluation threshold changed.

## Risks and Follow-ups

Fixed scale and event/right censoring only. Left/interval censoring, calibration,
IPCW quality, source licensing closure and complete A10 search remain open.
CPU/GPU transition prerequisites are recorded in the sprint; CUDA is unimplemented.

## Commits

- This A10 worker slice; parent 20d9b7c. Local only, no push.
