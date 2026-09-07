# Sprint 067 incremental execution evidence

[counts.json](counts.json) records clean revision `e99e89c` and exact synthetic
full replay for squared and Normal at 4/8/16/32 rounds. Counts are exactly 2*K*T:
one evaluation per new tree for training and validation. These are operation counts,
not time or peak memory. Historical quadratic evidence remains in runtime-audit-063.

[Installed checks](installed/manifest.json) rebuild the core and four development
extension wheels, exercise direct and M=1/8/32 scheduling, custom stopping and
ordered updates, then remove training plugins and exactly replay ten saved models
in fresh inference. This is internal conformance, not independent-author evidence.
All artifact and source hashes were independently verified against the revision.

Commands (with `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`):

- `uv run --no-sync python -m benchmarks.v1.runtime_cost_audit /tmp/openboost-counts-067.json`
- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-sprint067`

The unchanged frozen practical CPU diagnostic is running separately; no measured
speed or full-search gate is claimed by these artifacts.
