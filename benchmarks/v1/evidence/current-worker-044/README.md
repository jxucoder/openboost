# Current OpenBoost A1/A11 validation integration

All ten cells pass: two tasks on each of five frozen housing folds. Four rounds
per trial, depth two, 32 bins and patience three. This is plumbing, not a quality
search or a performance result. No test scores were computed.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-current-worker-044
```

Requires the pinned local housing archive and committed preprocessing freeze.
`summary.json` records revision/dirty state, source hashes, full jobs, runtime
identity, source and packet provenance, bounded worker outcomes and replay checks.
Each application/fold directory preserves raw validation predictions, JSON model,
training metadata, process evidence/log and fresh-process predictions. SHA256s
link these to the run. Generated training/validation/test packets are not copied;
their manifest hashes and deterministic export path permit reconstruction.

Validation predictions replay exactly; A1 outputs means, A11 outputs means and
positive standard deviations in original target units. Training selected the
strict best validation snapshot, independently of training-step acceptance.
The runtime supports CPU only. Memory was not capped; wall times are execution
records, not fair speed measurements. Packet separation is not OS label isolation.
This is partial M3 evidence, not F0.3/E3 or full delivery acceptance.
