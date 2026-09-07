# Current A7 count/exposure integration

All five frozen insurance frequency folds pass the current CPU worker and exact
fresh-process replay. Period counts and separate positive exposure vectors bind
to the public Poisson recipe. Exposure enters likelihood once; the real packets
use unit sample weights. Prediction requires exposure and returns period count
means, preserving source policy IDs.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-count-054 --applications A7
```

Requires pinned source files in build/v1-data and the committed preprocessing
freeze. The exporter validates all source, group split and encoded fold hashes.
Four rounds, depth two, 32 bins, learning rate 0.1, patience three, one CPU thread;
90-second worker and 30-second replay caps. Source export is outside fit timing.
No test labels scored, profiler, concurrent regression suite, CUDA or memory cap.
Host: Apple M4 Max, 51539607552 bytes physical RAM (same host as Sprint 052);
Python reports x86_64. Peak memory was not measured.

`summary.json` retains source/revision/dirty state, dataset and packet identities,
commands, environment and all outcomes. Each fold retains raw model, predictions,
replay, training metadata, process record and log. All source/output hashes match.
`verification.json` retains independently recomputed mean Poisson NLL including
log-factorial terms, math.fsum checks and exact source IDs. The maximum absolute
NLL discrepancy is approximately 2.4e-14 across different reduction/exp-log paths,
consistent with floating-point rounding. An initial rtol=1e-13 check failed on
fold two; retained final checks use rtol=1e-12 and atol=1e-14. Replay stays exact.

These are integration outcomes, not a complete search, calibrated quality
acceptance or comparative performance claim. Weighted synthetic tests separately
verify sample weights and exposure are applied independently.
