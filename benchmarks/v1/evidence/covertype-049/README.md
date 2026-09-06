# Full Covertype worker timeouts

All five frozen folds timed out at 90 seconds. No models or predictions exist;
probability and fresh-inference checks did not execute. This is failed validation
integration evidence, not a passing A3 result or a comparative speed claim.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-covertype-049 --applications A3
```

Requires pinned Covertype source and preprocessing freeze. Four rounds, depth
two, 32 bins, seven classes, one CPU thread. The summary retains source/revision,
data/packet/fold identities, exact jobs and commands, environment and all five
execution outcomes. Each fold retains its execution record and empty worker log.
Large generated packets are not copied; their frozen hashes and export procedure
are retained. No test labels were scored. CPU source hashes match the recorded
revision; the dirty state consists of planning records created before the run.

No memory cap or CUDA. Timeouts alone do not identify a hotspot or prove a
mathematical failure. Next bounded phase/stack profiling should use the same
full input before changing code or budgets.
