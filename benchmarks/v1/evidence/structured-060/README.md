# Current A12 structured saturation integration

All five frozen concrete folds pass the public Formula worker and exact fresh
replay. Encoded composition predictors feed parameter trees; age in days divided
by 28 is a separate positive structure role. Ordinary GBDT packets retain age as
a feature. Dedicated formula-input packets remove that appended column only.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --offline --no-sync --with xlrd python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-structured-060-retry --applications A12
```

The initial run without xlrd failed during source export before any fit. Its
failure and the retry dependency version/CLI are retained in verification.json.
The retry used cached xlrd in an ephemeral uv environment, without modifying
project dependencies. Pinned concrete source and preprocessing remain unchanged.

Four rounds, depth two, 32 bins, learning rate 0.1, patience three, public Formula
backtracking defaults and one CPU thread; unchanged 90/30-second fit/replay caps.
No test scoring, profiler, concurrent regression, memory cap or CUDA. Host: Apple
M4 Max, 51539607552 bytes RAM (same host as Sprint 052); Python reports x86_64.
Peak memory is unmeasured; export is outside fit timing.

Summary retains revision/dirty state, source/data/packet hashes, commands,
environment and results. Each fold retains raw model/predictions/replay/training,
execution records and log. All hashes match. Independent softplus/saturation and
weighted half-squared-error calculations match outputs/scores; age separation
and source IDs match exactly. This is integration, not extrapolation or A12
quality/search acceptance, baseline parity, or a performance claim.
