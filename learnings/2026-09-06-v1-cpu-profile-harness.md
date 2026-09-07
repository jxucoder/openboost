# 2026-09-06: Freeze the practical CPU diagnostic

## Context

Sprint 066 resource checks pass at `5243020`; practical fit costs remain unknown.

## Decision or Result

Freeze all eight cases and profile selection before execution. Preserve failures
and stop the remaining fit sweep after the first non-pass. Profile measurements
are separate from uninstrumented end-to-end timing.

## Changes

- Add a wheel-installed Modal coordinator and single-process worker with exact
  full-model replay, persistence replay, independent metrics and trace-byte counts.
- Freeze original Housing features and training/validation prefixes in the
  [protocol](../benchmarks/v1/evidence/practical-cpu-066/protocol.json).
- Test malformed inputs, row isolation, replay, metrics and forced rejection.

## Verification

Ruff passes; full CPU regression suite: **951 passed** (including diagnostic tests).
Actual numerical thread inspection is deliberately stubbed in source mathematics
checks because local NumPy exposes no pool. The remote worker requires observable
two-thread pools; source tests do not certify that environment.

## Failed Attempts

Local pool inspection initially failed two source tests. Keep optional threadpoolctl
loading lazy and use a stdlib density oracle so test collection needs no new package.

## Risks and Follow-ups

No practical measurements yet. Run the committed protocol with actual-image
resource checks; retain all raw results. No production optimization in this slice.

## Commits

- Profiling harness and preregistration; parent `5243020`.
