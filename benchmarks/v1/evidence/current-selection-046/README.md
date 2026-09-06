# Current synthetic A6/A13 selection integration

All 16 configurations pass; the independent mean-standardized-RMSE audit selects
openboost:15. The receipt is sealed and re-audited before releasing test features
for fresh-process selected-model inference. No test labels are scored.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.current_selection_smoke /tmp/openboost-selection-046
```

The seeded synthetic arrays, training-scale JSON, protocol, trial jobs, raw model
bundles, validation predictions, process records, receipt and final predictions
are retained. Summary records source/revision/environment identities, commands
and result hashes; protocol/records bind the full search artifacts. These local
paths describe the original run and can be regenerated in a new empty directory.
Training targets are explicitly bound to training row IDs and the frozen scale.
The constant output channel stays exactly seven after restoration.

This is internal synthetic integration evidence, not a real quality grid, speed
result, formal D5/E5/E7 evidence or OS-enforced label isolation. Other application
adapters, real searches and final A6 quality aggregation remain open.
