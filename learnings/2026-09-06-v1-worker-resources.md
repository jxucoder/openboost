# 2026-09-06: Verify the numerical worker before profiling

## Context

User resumed Sprint 066 after `e3af6c9`. The previous inspection could not observe
Modal gVisor cgroups. Its child address/deadline checks passed, but neither the
actual 8-GiB worker ceiling nor peak-memory response had been exercised.

## Decision or Result

Use a separately declared 8-GiB per-worker RLIMIT_AS ceiling for the single-process
diagnostic, retaining requested Modal caps and their unobservable status. This
constrains address space more conservatively than resident memory; report both
domains separately. Guest ru_maxrss responds to and retains a known touched
allocation, while VmHWM remains unavailable. Loaded OpenBLAS reports two threads.

## Changes

- Add an explicit numerical-worker probe mode while preserving original container
  inspection behavior and its failed artifact. Exercise exact 8-GiB limits, an
  oversized virtual mapping, a touched 64-MiB allocation and two-thread NumPy work.
- Preserve source/image/environment hashes and all probe outcomes in
  [worker evidence](../benchmarks/v1/evidence/worker-resources-066/README.md).
- Scope the diagnostic to one numerical process with threads and no spawned
  training workers. Do not generalize the bound to aggregate subprocess memory.

## Verification

With `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`:

- `uv run --no-sync python -m benchmarks.v1.resource_preflight /tmp/openboost-resource-worker-066.json --modal --worker`: all nine checks pass, exit 0.
- Independently verify the exact source hash, 8-GiB soft/hard limits, two-thread
  pool metadata, touched allocation and retained ru_maxrss response from raw JSON.
- Ruff, documentation links and staged diff checked before commit. No production
  code changed; no additional regression-suite or model-performance claim.

## Failed Attempts

No new failed worker probe. The earlier cgroup inspection failure remains unchanged.
Its premise was too specific to one host interface; missing files did not disprove
enforcement. VmHWM remains null rather than being substituted with zero memory.

## Risks and Follow-ups

Guest-reported high-water RSS is an observed counter, not independent hardware
instrumentation. A virtual ceiling can reject a workload before its RSS reaches
8 GiB; failures must retain this distinction. Next freeze the eight Housing cases,
repeat resource checks in the actual profiling image, and retain every failure.
Profile before choosing an incremental-runtime remedy. No GPU work is authorized
by this CPU resource result, and all formal gates remain unchanged.

## Commits

- Numerical worker resource verification; parent `e3af6c9`.
