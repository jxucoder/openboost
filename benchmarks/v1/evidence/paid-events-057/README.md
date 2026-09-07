# Matched positive-payment input binding

All five A9 training/validation folds bind to matched positive-payment counts,
totals and exposure. Independent claim iteration exactly reproduces the source
reader's bincount aggregates. Frozen source array hashes, original worker packet
hashes and training-ID hashes match. Source policy IDs map in exact frozen order;
eligibility, annualized targets and exposure weights are checked before output.

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.paid_event_data /tmp/openboost-aggregate-056/summary.json /tmp/openboost-paid-events-057
```

Regenerate Sprint 056 packets first if absent. The parent summary digest in
manifest.json matches the committed aggregate-056/summary.json. Pinned insurance
files are read from build/v1-data. Large generated feature packets are not copied;
the manifest retains their hashes, row/event counts and reconstruction command.
It also records revision/dirty state, source implementation hashes, source audit,
Python/NumPy/OS and input identities. No performance or memory measurement.

Only training/validation feature and matched aggregate roles are emitted. No
test label packet is opened; the preparer reads the full underlying source.
This is input binding, not a composition fit,
independent author result, quality/search acceptance or an OS access boundary.
