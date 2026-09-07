# Matched frequency-severity execution and replay

All five Sprint 057 hash-bound policy packets pass combined Poisson/Gamma fits
and exact fresh-process replay. Frequency uses matched positive-payment counts
with exposure; severity uses positive policy averages weighted by paid count.
No business weights or offsets are added. Separate component validation selects
models; the recorded joint_selection flag is false.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.composition_smoke /tmp/openboost-paid-events-057/manifest.json /tmp/openboost-composition-058
```

Regenerate Sprint 057 packets if absent. Binding hash matches its committed
manifest. Four rounds per component, depth two, 32 bins, learning rate 0.1,
patience three and one CPU thread. Both component fits share the 90-second cap;
fresh inference has 30 seconds. No profiler, concurrent regression, memory cap,
test labels or CUDA. Host: Apple M4 Max, 51539607552 bytes RAM (same host as
Sprint 052); Python reports x86_64. Peak memory was not measured.

The summary retains revision/dirty state, source and input hashes, exact commands,
environment, statuses and component stopping/model identities. Each fold retains
raw two-model persistence, all five named prediction arrays, replay, training,
execution and log records. All source/output hashes match. All arrays replay
exactly, preserve policy IDs, and are finite and positive. Products and exposure
conversions pass at rtol=1e-12. This does not establish joint aggregate selection,
full A9 quality/search, a compound distribution or comparative performance.
