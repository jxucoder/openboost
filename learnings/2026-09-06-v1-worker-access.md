# 2026-09-06: Probe a real worker/evaluator permission boundary

## Context

Frozen-manifest integrity does not establish permission isolation. The current
process runner bounds wall time but inherits evaluator identity/environment and
relies on external memory enforcement. Sprint 070 needs actual failed reads/writes
and resource failures before broader execution.

## Decision or Result

Add opt-in Linux hard address limits and a privilege-dropped worker path to the
existing runner. Keep parent evaluator ownership of private inputs, worker ownership
of output, and a minimal environment. No new universal executor. Unsupported local
macOS use is rejected explicitly. Actual Linux/Modal evidence follows clean commit.

## Changes

- Explicit RLIMIT_AS, UID/GID/no_new_privs and empty group setup via an exec wrapper;
  no preexec_fn in the potentially threaded coordinator.
- Same-process-group cleanup before reading successful worker artifacts.
- Bounded synthetic permission/resource probe, including failed parent-environment
  access and real allocation/error/deadline cases. No real sealed inputs uploaded.

## Verification

Eight focused process-runner tests pass. Five new tests initially fail on absent
limit/identity controls, then pass after implementation. Full regression: **1018 passed**; lint and
docs pass before commit. Local tests do not certify the Linux privilege transition.

## Failed Attempts

The clean Modal run at `26a6797` passes all 13 checks: one numerical success and
six expected process failures. No retry or preemption was reported. Keep requested
container memory distinct from the verified process address-space limit.

## Risks and Follow-ups

This is not a complete hostile-code sandbox. New sessions, network and sibling
same-UID workers need separate container/attempt isolation. The probe configures
1800 seconds for a quick numerical success and exercises a short forced timeout;
it does not observe an 1800-second expiry or prove full model fits meet that budget.
Keep all failed/partial outcomes and narrow claims to checks actually exercised.
Full-search and independent author accounting remain separate unfinished work.

## Commits

- Runner controls and preregistered real probe; parent `acac36a`.

## Committed evidence checkpoint

[Raw evidence](../benchmarks/v1/evidence/worker-access-070/README.md) retains seven
execution records and all logs. Independent local inspection verifies 17 artifact
hashes and three committed source hashes. Actual protected-path operations return
PermissionError, 9-GiB mapping returns ENOMEM, and the short timeout exits -9 with
its partial log retained. Numerical identity/privilege/environment assertions pass.
The touched 64-MiB allocation verifies ru_maxrss accounting; proc high-water data
is unavailable. These results do not qualify full search or independent authors.
