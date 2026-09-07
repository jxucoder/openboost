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

### Launch correction

Modal rejects any explicit retry parameter on generator functions, including zero.
Remove that parameter; generator execution has no retry policy. The failed launch
ran no remote work and is retained as `launch-error.json`. Retry uses identical
frozen input/protocol in a fresh directory. Ruff and the ten focused profile tests
pass before the correction commit.

The launch correction also accidentally omitted the retry declaration from future
freeze metadata. Restore that field and add a regression assertion. The active
run uses the original committed protocol, which retained `retries: 0` throughout;
its execution and inputs are unaffected. Ten focused tests and format checks pass.

### Preemption recovery amendment

Modal preempted the sweep after six passing cases, during squared 8192/128, and
announced infrastructure restart despite no application retry policy. Stop the app
immediately; retain the six streamed cases and mark the interrupted case separately
from a timeout. Normal 8192/128 remains not_run. No timing/RSS can be recovered for
the interrupted child. Infrastructure recovery is not controlled by `retries`.

Add an explicit profile-only recovery option selecting an existing frozen case.
Before launching recovery, select squared-8192-128 with the original 60-second soft
and 120-second hard deadlines. Do not repeat any uninstrumented fits. Reject a
second preflight event to stop observed generator restarts before repeated fits.
This detects a restart after its preflight, not preemption prevention or a guarantee
against all infrastructure compute repetition. Retain the operational limitation.
