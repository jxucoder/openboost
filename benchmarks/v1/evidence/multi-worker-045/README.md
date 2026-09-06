# A6 current worker integration

Five grouped Parkinsons folds pass using four-round shared multi-output trees,
32 bins, patience three and one CPU thread. Saved scale metadata exactly equals
the frozen unweighted training-population scale. Fresh-process original-unit
predictions match exactly. No test scores, search or performance claim.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-multi-worker-045 --applications A6
```

Requires the pinned local data and committed preprocessing freeze. The summary
records source/revision/environment/input identity, exact jobs, model/validation
artifact hashes and process outcomes. Each fold retains raw predictions, JSON
model and scale, training state metadata, process record/log and exact replay.
Generated source packets can be reconstructed with worker_data.export; hashes
are retained in the summary. Independent-tree mode has synthetic unit coverage,
not real five-fold evidence in this directory. No memory cap or CUDA was used.
