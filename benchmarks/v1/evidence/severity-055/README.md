# Current A8 positive-claim severity integration

All five frozen policy-grouped severity folds pass the current CPU worker and
exact fresh-process replay. Targets remain individual eligible positive joined
claims, with unit claim weights in the real packets. Output is positive claim
mean; no exposure or policy-average substitution is applied.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-severity-055 --applications A8
```

Requires pinned insurance source files in build/v1-data and the committed
preprocessing freeze. The exporter verifies source, grouped population, row and
encoding hashes. Four rounds, depth two, 32 bins, learning rate 0.1, patience
three and one CPU thread; 90-second fit/30-second replay caps. Source export is
outside fit timing. No test labels scored, memory cap, profiler or concurrent
regression suite. Host: Apple M4 Max, 51539607552 bytes RAM (same host as Sprint
052); Python reports x86_64. No CUDA or measured peak memory.

`summary.json` retains source/revision/dirty state, dataset/packet identities,
commands, environment and results. Raw models, predictions, replay, training,
execution records and logs are retained for every fold. All source/output hashes
match. `verification.json` records independently recomputed Gamma objectives
(y/mean+log(mean)), positive means and exact source IDs. Numerical comparisons
use rtol=1e-12 and atol=1e-14; prediction replay is exact.

These are bounded integration outcomes, not full quality searches, dispersion
estimation, calibration or comparative performance results. Synthetic weighted
checks separately verify original sample-weight semantics.
