# Current classification worker integration

`passed/` records 5/5 frozen Adult folds passing after the packet-ID adapter fix.
`failed/` retains all five original failures: NumericData rejected external string
IDs. Its worker-source.txt matches the exact recorded failing source hash.
The passing worker validates/preserves source IDs in outputs and uses local integer
indices internally. Exact fresh-process probabilities and IDs match.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-classification-048-fixed --applications A2
```

The pinned Adult archive and preprocessing freeze are required. Summaries retain
source/revision/environment/input identity, exact jobs and output hashes; fold
directories preserve available models, predictions, training and execution records,
logs and replay. Generated packets are reproducible from their recorded export
hashes. Historical failure hashes describe their original dirty source state.

Four rounds, depth two, 32 bins, one thread. No test scores, quality/calibration,
search or speed claims. A3 has synthetic weighted/direct/fresh checks; full
Covertype runs remain pending. No memory cap or CUDA was used.
