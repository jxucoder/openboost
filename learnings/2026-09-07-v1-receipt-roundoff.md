# 2026-09-07: Bound score roundoff during receipt replay

## Context

The unchanged committed Linux receipt rejects macOS replay although all artifact
hashes, non-score fields and the winner agree. Twelve score values have last-bit
differences. The raw counterexample remains unchanged in protected-selection-070.

## Decision or Result

Permit differences only in finite numeric score leaves, bounded by eight times
the smaller binary64 spacing. Require exact dictionary structure and non-score
fields. Recompute the winner and require the stored scores to select that same
winner. Never round scores before selection or treat close winners as equivalent.
The raw receipt byte hash, protocol, records and artifacts remain exactly pinned.

## Changes

- Narrow numerical comparison during release, after fresh independent audit.
- Regression directly replays the original committed Linux artifact.
- Boundary, malformed/material score and near-tie reversal coverage.

## Verification

Three initial failures reproduce one/eight-spacing and real Linux replay rejection.
All 30 focused selection tests pass after implementation. Existing modified model,
receipt, configuration and forged-winner cases still reject release.

## Failed Attempts

The first default-parallel full run has one unrelated two-second process startup
timeout (1030 passed, one skipped). Re-run with two test workers to check under
bounded contention; do not raise the process budget or change that test here.

## Risks and Follow-ups

Eight spacings is a deliberately narrow supported replay bound, not a universal
floating-point error theorem. Larger differences, changed ranking and other exact
numerical contracts remain fail-closed. In particular, target-scale recomputation
has not been relaxed. Full search and formal quality/author/device gates remain open.

## Commits

- Numerical receipt contract; parent `39aa31d`.

Final verification with two test workers: **1031 passed, 1 skipped** (Linux-only).
Production/support lint and documentation build pass. No new Modal run was needed
to replay the unchanged Linux bundle on the host that exposed the counterexample.
