# Full Covertype replay after histogram gather reuse

All five unchanged full-data jobs pass their original 90-second process cap:
folds 0–4 take 70.3, 69.9, 68.1, 67.8 and 67.9 seconds respectively. Each saved
model reproduces validation probabilities and source row IDs exactly in a fresh
process under the 30-second replay cap. Fold zero's model bytes and prediction
arrays also exactly match Sprint 051. This is bounded validation integration,
not a quality search, matched-quality speed comparison or A3 acceptance.

`summary.json` retains revision/dirty state, source hashes, jobs, packet hashes,
commands, environment and execution outcomes. `input-summary.json` retains the
original Sprint 049 source/version/split/packet provenance and failed outcomes;
its bytes match the prior-summary digest. Per-fold raw models, predictions,
replays, training metadata, execution records and logs are retained.
`verification.json` records hash/equality checks and supplemental CPU/RAM metadata.

Regenerate the frozen packets with the Sprint 049 export procedure if absent,
then rerun from that generated summary into a fresh output directory:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.replay_current_packets /tmp/openboost-covertype-049/summary.json /tmp/openboost-histogram-052
```

Four rounds, depth two, 32 bins, seven classes and one CPU thread remain unchanged.
Source export is outside worker timing. No profiler or concurrent regression
suite ran during these fits. No memory cap, measured peak memory, CUDA or test
label scoring. The implementation retains a contiguous selected-statistic buffer;
transpose construction can transiently hold two rows-by-fields arrays. No memory
improvement or stable speed ratio is inferred from these local reruns.
