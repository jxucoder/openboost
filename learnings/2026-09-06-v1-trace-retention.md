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

### Measurement harness

Extend the existing frozen CPU coordinator with explicit same-wheel full/summary
pairing over all eight cases, followed by two summary profiles. Per-worker and
aggregate caps remain unchanged; the sixteen-fit/two-profile amendment is recorded
before execution. Worker outputs retain trace mode, completed records and logical
array bytes separately from guest RSS. Add a focused summary-worker assertion and
extend the synthetic count audit with an explicit retention option. No runtime
change in this measurement slice. Eleven focused profile tests and lint run before
commit; installed extension execution is recorded separately.

### Clean installed and count evidence

Installed verification at `2f5aca2` passes the earlier author/scheduler checks,
ten plugin-free inference models and explicit full/summary mixed M=1/8/32 cases.
At `ae8c7af`, full and summary count audits preserve exact replay and 2*K*T tree
work; summary step arrays are zero for all eight counting cases. Source/artifact
hashes verified. See [retention-068](../benchmarks/v1/evidence/retention-068/README.md).
This is logical-retention evidence, not yet an RSS result.

### Paired measurement outcome

All eight full/summary pairs pass with exact predictions and byte-identical model
JSON. Summary retains every round record and zero per-round array bytes. At
8192 rows/128 rounds, guest peak RSS is 103.320 → 88.500 MiB for squared and
156.887 → 93.441 MiB for Normal. Full trace arrays are 16.0625/64.125 MiB.
Some small squared cases have slightly higher summary RSS; preserve these observations
instead of claiming uniform savings. Single ordered pairs are not repeated memory
or timing estimates. Final CPU regression now passes 1000 tests, including the
additional summary-worker test. There is no training-semantic or tolerance change.

Decision: the identified trace-retention blocker is resolved. Do not declare the
full-search workload qualified from this smaller diagnostic. Sprint 070 must audit
and enforce the exact full-search environment and record a bounded preflight before
071 expansion. Existing real workers still need explicit summary selection rather
than silently inheriting it: full remains the public default. Sprint 069 preparation
can proceed; independent attempts require the authorization and isolation in its card.

### Closure verification

Both final summary profiles pass, preserving all 128 records, zero step-array bytes
and 256/512 tree calls. All sixteen fit metrics and both profile metrics were
recomputed independently; wheel/source/loaded-module/artifact hashes and exact
paired models verified. Evidence is committed under retention-068/practical.
Ruff, docs and package checks pass; raw profile text retains its hash-preserving
trailing blank line. Close 068 and return to 069/070 readiness work rather than
adding unplanned CPU optimization. No remaining blocker within this bounded sprint.
