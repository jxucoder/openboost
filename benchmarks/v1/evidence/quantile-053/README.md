# Current A5 quantile integration on frozen Bike origins

All five frozen calendar-only Bike rolling origins pass the current OpenBoost
CPU worker and exact fresh-process inference. Each job composes three independent
scalar quantile recipes at 0.1, 0.5 and 0.9, with four rounds, depth two, 32 bins,
learning rate 0.1 and patience three. Selection and stopping remain per quantile.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-quantile-053 --applications A5
```

Requires the pinned archive at /tmp/openboost-v1-bike.zip and committed preprocessing
freeze. The exporter checks source hashes, calendar-only features, train-fitted
encoding and whole-date chronological prefixes. Later origin rows are unused.
The worker receives training and validation only; no test labels were scored.

`summary.json` retains revision/dirty state, source hashes, source/split/packet
identities, exact jobs/commands, environment, stopping and output metadata. Every
fold retains its raw model, predictions, replay, training record, process record
and log. `verification.json` records independently recomputed selected validation
pinball scores and exact source-row alignment. All source/output hashes match.
Supplemental host metadata is Apple M4 Max, 51539607552 bytes physical RAM
(the same host as Sprint 052); Python reports x86_64. One process thread, 90-second
fit cap, 30-second replay cap; no memory cap, peak-memory measurement or CUDA.

No observed crossings in these short runs does not imply a noncrossing guarantee.
No sorting is applied. This is validation plumbing, not a 16-trial search,
calibration/quality acceptance or performance comparison. Source export is outside
fit timing, and a small focused test ran during the integration invocation.
