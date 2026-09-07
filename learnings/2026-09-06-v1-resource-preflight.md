# 2026-09-06: Resource observability is distinct from requested limits

## Context

Sprint 066 starts after installed-isolation commit `e2b6c4c`. The user requested
continued execution, frequent commits and a later retrospective checkpoint.
The existing process runner explicitly leaves memory enforcement to its host;
the new diagnostic requires a verified budget before fitting any Housing cases.

## Decision or Result

The local macOS attempt to set a 128-MiB RLIMIT_AS failed with `current limit
exceeds maximum limit` before the attempted allocation. An authorized single
Modal CPU preflight then verified allocation failure, hard timeout killing,
partial logs and normal completion. Container cgroup inspection was unavailable.
The complete preflight remains failed/incomplete, with raw failure preserved.

Missing cgroup files are not evidence of missing host enforcement. RLIMIT_AS is
a virtual-address ceiling, not peak RSS. These distinctions must remain visible
before an end-to-end resource or speed claim. No profiling fits were launched.

## Changes

- Add a Linux child executor with an address-space limit and a parent-enforced
  process-group deadline. The small probe uploads only its own script, with no
  source checkout, dataset or secret included through automatic source capture.
- Record requested hard resource limits, actual inspection values, all child
  commands/status/logs, source hash, image and environment in
  [raw evidence](../benchmarks/v1/evidence/resource-preflight-066/README.md).
- Record the failed preflight and a bounded follow-up in Sprint 066, preserving
  all eight planned Housing cases as not_run.

## Verification

Commands use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.

- `uv run --no-sync python -m benchmarks.v1.resource_preflight /tmp/openboost-resource-preflight-066.json --modal`:
  expected nonzero overall result. Allocation exits 73 with MemoryError, timeout
  exits -9 with the initial log, normal child exits 0. Both cgroup paths are null.
- Independently checked the committed script against the recorded SHA256, the
  four passing child checks, both unavailable inspection values and overall failure.
- `uv run --no-sync ruff check benchmarks/v1/resource_preflight.py`: passed.
- Local documentation targets and staged diff checked at commit closure. No core
  change or new full-suite claim; latest CPU regression is 943 passes from 064.

## Failed Attempts

macOS rejected the attempted address-space limit. The first Modal probe incorrectly
assumed cgroup-v2 limit files were inspectable in this gVisor environment. Keep
that result rather than change false flags into passes. The child enforcement
checks themselves behaved as expected and retained failure evidence.

## Risks and Follow-ups

At this third implementation/evidence commit, pause for the requested retrospective:
the common completion boundary is smaller and installed semantics are verified;
the next risk is practical execution and its measurement, not more expressiveness.
Next verify an actual 8-GiB per-worker address bound, usable peak-RSS accounting
and two-thread numerical execution on Modal before the frozen eight-case matrix.
Explain any stricter address-space constraint without rewriting the existing RAM
budget or claiming aggregate control over arbitrary subprocesses. No CUDA gate,
formal author comparison, selected quality or adoption evidence was added.

## Commits

- Resource preflight and checkpoint; parent `e2b6c4c`.
