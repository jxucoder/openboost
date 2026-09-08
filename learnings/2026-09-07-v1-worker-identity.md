# 2026-09-07: Verify worker identity across exec

## Context

[096's real smoke](../v1-sprints/096-linux-worker-result.md) ran as root because
the provider ignores Dockerfile `USER`. Five checks failed, including actual core
and material writes. The user approved the explicit process correction and a
separate bounded corrected validation. Original failed evidence stays immutable.

## Decision or Result

Use a trusted entrypoint to set `no_new_privs`, clear supplementary groups and set
real/effective/saved GID and UID to 1000. Check these values before and after a
fresh interpreter exec, before executing any worker code. Each future author
command must use this boundary. A provider accepting image construction does not
verify runtime identity; actual Linux execution remains a distinct gate.

## Changes

- `linux_launcher.py`: explicit privilege reduction and inherited-identity guard;
  failed syscalls, incorrect identity or unsupported hosts stop before worker code.
- `modal_worker.py`: launcher precedes the only smoke command, image no longer
  requests ignored `USER`, and the classifier requires both identity records plus
  the original nineteen checks and actual timeout. The 096 classifier is retained.
- [097 plan](../v1-sprints/097-explicit-worker-identity.md) and separate freeze:
  fourteen selected uploads, original packet/probe bytes, one 90-second CPU
  Sandbox with two CPUs and 2048 MiB, zero retries, no model or GPU.

## Verification

- First failing test: `test_worker_command_cannot_bypass_privilege_launcher`
  raises `AttributeError` before construction; no guarded command existed.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/v1/test_author_worker_identity.py tests/v1/test_author_linux_worker.py -n 0 -q`:
  52 pass. This includes installed-SDK image construction, actual macOS rejection,
  syscall failure/order checks, saved-root/group/privilege rejection, inherited
  guard control flow and replay of the unchanged original failed evidence.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  1880 pass, one Linux-only skip in 13.69 seconds. Ruff passes for production and
  all four changed Python files. These local checks do not verify real Linux
  privilege enforcement, author cost or provider timeout.
- The local freeze CLI passes: 24 frozen inputs and fourteen uploads; the original
  thirteen delivery hashes and nineteen cases are unchanged. All 85 pending run-8
  source hashes match. MkDocs builds with the existing `execution.md` link warning;
  authored-file whitespace checks pass. No service access occurred during these
  checks.

## Failed Attempts

The original nineteen cases and raw 096 failure are unchanged. Mocked syscall
results test failure ordering only; they cannot replace the approved real run.

## Risks and Follow-ups

Execute the separate corrected freeze once and retain every result, then reflect.
Full author time/token enforcement, fair arms/model/settings and actual independent
authoring remain open. The numerical fixtures and pending GPU run-8 sources stay
unchanged. Do not expand isolation fixtures indefinitely instead of returning to
the foundation's author-benefit question.

## Commits

- Parent `3eb37f6` retains the failed 096 smoke and retrospective.
- This implementation commit contains the correction and approved 097 freeze.

## Actual Linux result and reflection

The corrected smoke executes once at clean `518eccf`: all nineteen original checks,
both identity guards and actual provider expiry pass. Real/effective/saved UID/GID
are all 1000, supplementary groups are empty, and `no_new_privs=1` survives exec.
Core/material writes return `EACCES`, root restoration returns `EPERM`, and the
core plus four controller evaluator hashes remain unchanged. All public examples
work. Every original delivery byte and case is preserved; only the launcher was
added. [Raw evidence](../benchmarks/v1/evidence/author-linux-identity-097/README.md)
retains all eighteen artifacts and the exact approved freeze.

The provider timeout occurs after the separate-session child marker; return code
124 and final-parent-sleep `KeyboardInterrupt` are retained. Creation including
image work takes 22.540 seconds; the wait phase takes 90.949 seconds. Cleanup returns
without error. No independent post-expiry child-PID observation or full author
time/token enforcement is claimed. The active freeze changes only authorization
to `consumed`; no retry, model generation or GPU invocation occurs.

Local replay accepts the actual result only with timeout, while the original 096
failure still fails. All 21 indexed archive hashes and all 24 execution-source
hashes match; the 85 pending GPU run-8 source hashes remain unchanged. This measured
correction closes the declared worker smoke. It does not measure algorithm-change
cost or pass independent-author gates. The [retrospective](../v1-sprints/097-worker-identity-result.md)
stops further sandbox expansion and returns to a real token/time controller and
fair-arm/model/settings design before an authorized author attempt.

Closure verification also confirms that the consumed CLI rejects locally before
creating output. MkDocs builds with the same existing evidence-link warning;
authored-file whitespace checks pass. The raw builder log deliberately retains
its original trailing spaces and final blank lines. Stage those exact bytes and
the ignored wheel explicitly. All 21 indexed hashes were verified against staged
Git blobs, and all 23 archive files are staged. No production source changes
after `518eccf`.
