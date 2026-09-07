# 2026-09-06: Practical CPU replay diagnosis

## Context

Sprint 066 tests whether the tiny quadratic replay counterexample matters on
frozen Housing input before changing the public transaction runtime.

## Decision or Result

Six uninstrumented fits pass exact replay and independent validation metrics.
The separate 60-second partial squared profile spends 75.3% in tree prediction,
including 67.5% re-encoding; NumPy searchsorted self time is 50.5%. Proceed with
Sprint 067 incremental evaluation and identity-bound encoding reuse. Keep exact
summation order and public isolation guarantees. Trace retention is not the
observed primary CPU bottleneck; its separate sprint remains later.

## Changes

- Commit raw sweep and recovery artifacts, source/protocol hashes and all statuses
  under [practical CPU evidence](../benchmarks/v1/evidence/practical-cpu-066/README.md).
- Update the execution handoff and Sprint 067 with measured scope and baseline gaps.

## Verification

951 CPU tests pass at harness introduction. Subsequent coordinator-only corrections
pass ten focused tests, Ruff and formatting. Documentation build passes. Independently
verify every copied case artifact hash, source against recorded revision, wheel hash,
row IDs and squared/Normal metrics from predictions and original frozen targets.
Recovery source hashes and artifacts also verified. Core source unchanged.
Staged diff whitespace checks pass except the verbatim pstats text report
trailing blank line, deliberately retained to preserve its raw hash.

## Failed Attempts

The initial launch rejected an explicit retry parameter on a generator. A later
container preemption interrupted squared 8192/128 and announced infrastructure
restart. Stop the app; preserve six results. The original manifest remains error,
with a separate interruption record covering all cases. Normal 8192/128 is not_run.
Recovery profile hits its planned soft deadline, retaining partial pstats and JSON;
no final fit metric or peak RSS is inferred. No uninstrumented fit repeated.

## Risks and Follow-ups

One timing observation per completed case, no full long-round baseline and no
long-round Normal profile. Cumulative profile times overlap. Guest RSS is not host
hardware instrumentation; 8-GiB address limit is more conservative than resident
memory. Generator retry settings do not prevent infrastructure recovery. A repeated
preflight is detected locally but that is not a no-preemption guarantee.

At the requested retro checkpoint, stop before editing production runtime. Next
067 requires exact full replay and broad transaction/ownership conformance, then
the same frozen diagnostic. Formal quality, cost, author, CUDA and adoption gates
remain open; no push or external performance claim.

## Commits

- Harness `fad659b`; launch correction `f5ca13b`; freeze metadata `c96f7af`;
  profile-only recovery `dff0590`; evidence closure follows these revisions.
