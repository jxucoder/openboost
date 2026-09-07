# Current A9 direct annualized aggregate integration

All five frozen aggregate folds pass the current CPU Tweedie worker and exact
fresh-process replay. Annualized paid totals use positive exposure weights once,
with fixed power 1.5 and no extra offset. Output retains annualized units.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-aggregate-056 --applications A9
```

Requires pinned insurance files in build/v1-data and committed preprocessing
freeze. Four rounds, depth two, 32 bins, learning rate 0.1, patience three and one
CPU thread; 90-second fit/30-second replay caps. No test labels scored, memory
cap, profiler or concurrent regression. Export is outside fit timing. Host:
Apple M4 Max, 51539607552 bytes RAM (same host as Sprint 052); Python reports
x86_64. No CUDA or measured peak memory.

`summary.json` retains source/revision/dirty state, population/split/packet hashes,
commands, environment and outcomes. Each fold retains raw model, predictions,
replay, training, execution and log artifacts. All source/output hashes match.
`verification.json` records independent weighted Tweedie objectives and exact
source-ID/exposure-weight agreement with the hashed validation-period packet.
Annualized targets times exposure reproduce paid totals. Additional raw
period-predictions.npz files retain annualized predictions times exposure.
Numerical checks use rtol=1e-12/atol=1e-14; fresh replay is exact.

This is direct aggregate integration, not a complete A9 search, calibrated
compound distribution or performance comparison. Frequency-severity composition
still requires matched positive-payment counts and totals; A7 raw claim counts
cannot be substituted. No required scope is dropped.
