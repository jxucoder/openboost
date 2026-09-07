# 2026-09-06: Explicit bounded trace retention

## Context

Sprint 067 removes repeated prediction work, but full traces retain 16.0625/64.125
MiB of logical arrays at 128 rounds on Housing for squared/Normal. This is retained
array evidence, not proof of a leak or a process memory limit failure.

## Decision or Result

Keep full diagnostics as the default and add opt-in summary retention to all twelve
built-ins. Convert each step before appending, retaining scalar diagnostics and
per-output MSE tuples while omitting sample arrays. Preserve validation and every
outer-round observation. Public TraceSummary accepts immutable scalar payloads;
external authors explicitly choose their summary and keep result ownership.

## Changes

- Public diagnostics record plus immediate built-in trace retention.
- Ordered development recipe explicitly retains scalar substeps/commit versions
  in summary mode, without past AcceptedState objects.
- All-family source equivalence, stopped/zero/rejected cases and installed mixed
  M=1/8/32 retention checks. Structural result validation is unchanged.

## Verification

Full CPU regression: **999 passed**, including 42 focused retention tests.
Ruff, docs and offline package build pass. Installed and practical measurements
follow a clean implementation commit; this slice makes no measured RSS/time claim.

## Failed Attempts

The initial counterexample rejects the nonexistent retention argument as expected.
No training algorithm or acceptance tolerance changed to make summary pass.

## Risks and Follow-ups

Summary stores per-output metrics, so O(rounds) assumes fixed output width and
trial budget. Current/raw arrays, best models, encodings and temporary work remain.
External payloads are not automatically pruned; authors must opt in explicitly.
Measure actual guest peak RSS separately and compare exact predictions. Follow
same-container comparison when CPU architecture differs; preserve baseline evidence.

## Commits

- Retention implementation; parent `dd2d696`.
