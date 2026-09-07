# Sprint 066 resource preflight: child limits work, container inspection unavailable

Parent `e2b6c4c` plus the dirty probe source hashed in preflight.json. One authorized
Modal CPU function requested two CPUs with a two-CPU hard limit, 8,192 MiB with
an 8,192-MiB hard limit, a 60-second function timeout and no retries. Only the
explicit probe script was uploaded. No data or model training was involved.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.resource_preflight /tmp/openboost-resource-preflight-066.json --modal
```

Use a fresh output path. The command exits nonzero because the complete preflight
did not pass. The JSON is retained, including exact commands, source hash, image
ID, Modal/Python/OS versions, requested limits, child logs and exit statuses.

| Probe | Observed result |
|---|---|
| Child tries a 256-MiB allocation under 128-MiB RLIMIT_AS | Expected MemoryError; child exits 73; diagnostic retained |
| Child exceeds one-second deadline | Process group killed with SIGKILL; exit -9; initial log retained |
| Valid child under the same address ceiling | Exit 0 |
| Inspect 8-GiB container cap through memory.max | File unavailable; value null; check cannot pass |
| Inspect two-CPU quota through cpu.max | File unavailable; value null; check cannot pass |

The environment reports Linux 4.19.0 gVisor and two CPUs. Missing cgroup files do
not demonstrate that Modal failed to enforce requested resources. The boolean
container checks mean this inspection did not verify them, not measured absence
of a limit. Likewise RLIMIT_AS constrains virtual address space, not a peak-RSS
measurement or a general aggregate bound for arbitrary child processes.

This reveals a mismatch between the chosen inspection mechanism and the runtime.
The next slice should verify an explicit per-worker 8-GiB address ceiling and
available peak-RSS accounting, recording virtual/resident scope separately and
retaining the original requested container caps. The current 128-MiB probe does
not itself establish the planned profiling worker's full 8-GiB acceptance.
Do not call this profile completion or proceed with an unenforced/advisory label.

No frozen Housing case ran. All eight remain not_run; no CPU performance, memory
improvement, application quality or phase pass follows from this preflight.
