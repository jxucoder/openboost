# 2026-09-05: P2 execution boundaries and baseline

## Context

P2.1 passed on T4 at 8609f70. Continue with execution boundaries and the planned
real-data baseline, preserving wheel-only provenance and prior artifacts.

## Decision or Result

Add a separate boundaries suite so historical correctness artifacts retain
stable required-test semantics. Eight new GPU cases join the five existing
ones: custom same-name distribution, exposure, generic tree fallback, runtime
error rollback, two sampling preflights, and two callback/eval/persistence cases.

The raw-score download spy checks a specific trainer boundary; it is not a
complete transfer counter. Existing tree conversion and evaluation may still
copy arrays to the host. No zero-transfer claim follows.

## Changes

- `tests/foundation/test_boundaries.py`: exercise actual CUDA execution and
  visible fallback; CPU comparisons; GPU-save/CPU-load and reverse round trips.
- Isolated runner/preparer: new boundaries suite, complete-case validation.

## Verification

- Host required-case tests and lint before commit; real GPU follows clean build.
- No production code changed in this slice.

## Failed Attempts

None yet.

## Risks and Follow-ups

- Run boundaries on real T4 before declaring P2.2 complete.
- Download the hash-pinned California Housing archive and freeze seed 0/1/2
  train/validation/test splits, then record Normal quality and cold/warm timings.
- Real-data baseline is not yet measured; do not advance P2 status prematurely.

## Commits

- `576702d` — P2.1 before/after T4 evidence.
