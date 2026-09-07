# Current A10 event/right-censored AFT integration

All five frozen Veteran folds pass the current CPU AFT worker and exact fresh
inference. Events become equal positive bounds; right-censored times become
positive lower/infinite upper bounds. Output is [log-time location, sigma] with
fixed sigma=1, matching the evaluation contract. No censoring-as-event substitution.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-survival-059 --applications A10
```

Requires pinned Veteran source in build/v1-data and committed preprocessing
freeze. Four rounds, depth two, 32 bins, learning rate 0.1, patience three, one
CPU thread and unchanged 90-second fit/30-second replay caps. No test scoring,
profiler, concurrent regression, memory cap or CUDA. Host: Apple M4 Max,
51539607552 bytes RAM (same host as Sprint 052); Python reports x86_64.
Peak memory was not measured. Existing source licensing closure remains open.

Summary retains revision/dirty state, source/data/split/packet hashes, commands,
environment and results. Raw model, predictions, replay, training metadata,
execution and log records are retained per fold. All source/output hashes match.
Verification records independently computed event density/right-survival NLL
using math.erfc, with rtol=1e-12/atol=1e-14, and exact source-ID agreement.
These are bounded integration checks, not IPCW/calibration, quality/search,
learned-scale, other-censoring or comparative performance acceptance.
