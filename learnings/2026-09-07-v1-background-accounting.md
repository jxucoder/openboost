# 2026-09-07: Cancellation and final usage are separate observations

## Context

[098](../v1-sprints/098-author-request-accounting.md) reserves output before one
supervised synchronous request. Client expiry alone cannot establish provider
stop or final generated usage. The next step toward 069 is a small real accounting
observation, preceded by a concrete cancellation path and request freeze.

## Decision or Result

Add optional background transport to the trusted controller. Permit only create,
retrieve and cancel on the fixed Responses host. A known response ID allows one
cancel plus one final retrieval within a shared 15-second cleanup interval after
interruption. Cleanup grants no author work. Valid terminal usage reconciles the
reservation, but the interrupted answer is never returned, even if completion
races cancellation. Unknown creation IDs or usage remain unresolved and closed.

The current official cancellation example has null usage. Therefore confirmed
cancellation and complete accounting are separate acceptance conditions; neither
local fixtures nor cancellation acknowledgement close the real usage gate.

## Changes

- `responses_transport.py`: fixed retrieve/cancel routes with validated response
  identifiers; the existing HTTPS subprocess, byte bound and no-retry behavior
  apply to every operation.
- `responses_background.py`: sequential polling and bounded cleanup, retaining
  each operation, raw reply and interruption. No new generation in cleanup.
- `accounting.py`: optional background mode, terminal reconciliation after a
  stopped transport and unconditional withholding of the stopped answer.
- [099 plan](../v1-sprints/099-background-accounting-smoke.md): construction,
  provider decisions and a separately reviewable live smoke to follow.

## Verification

- First new test fails to import the absent background implementation. It then
  verifies deadline-triggered cancellation, final retrieval and known usage while
  the caller receives a timeout, not a model answer.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_background.py tests/v1/test_author_accounting.py -n 0 -q`:
  43 pass in 5.41 seconds. Eleven new cases cover normal polling, cancellation,
  unknown usage/identity, completion races, cleanup failure and shared expiry,
  and HTTPS retrieve/cancel endpoint confinement. Counts are explicit fixtures.
- Ruff and formatting pass for the changed support files. No production code,
  dependency, frozen worker source or GPU run-8 source changes in this slice.

## Failed Attempts

An interruption with a cancelled response but null usage cannot reconcile the
reservation. The tests preserve this as unknown rather than zero-token success.
The real API has not been invoked; account access and actual cancellation behavior
remain unverified.

## Risks and Follow-ups

Freeze exact public text inputs, model alias/settings, maximum requests, output
caps, work/cleanup windows, cost estimate and verdict before requesting a live
allowance. Model generation, independent authors, worker integration and GPU run 8
remain pending. Abrupt controller/host loss can still prevent cleanup; a bounded
cancel attempt is not a guarantee of immediate provider cancellation.

## Commits

- `71ef69e`: preceding local request accounting.
- This slice adds local cancellation construction, not a provider result.
