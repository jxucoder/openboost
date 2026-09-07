# Sprint 068 retention evidence

The [installed manifest](installed/manifest.json) records clean implementation
`2f5aca2`: D1–D4/custom stopping/ordered checks and ten fresh inference models pass.
[Retention checks](installed/retention-checks.json) explicitly compare mixed
M=1/8/32 full/summary runs, including ordered author-owned payloads, unchanged
state/stop identities and reordered execution. Summary payloads contain no arrays
or AcceptedState objects. Structural result ownership remains unchanged.

[counts-full.json](counts-full.json) and [counts-summary.json](counts-summary.json)
record clean revision `ae8c7af` on the pinned 4/8/16/32-round fixture. Both retain
2*K*T tree evaluations and exact full replay. Summary retains zero step-array bytes;
full logical arrays grow with rounds. Scalar records and models still occupy memory;
zero array bytes does not mean zero trace memory or a measured RSS reduction.

Source and artifact hashes were independently verified. Commands use
`UV_CACHE_DIR=/tmp/openboost-research-uv-cache`:

- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-sprint068`
- `uv run --no-sync python -m benchmarks.v1.runtime_cost_audit /tmp/openboost-counts-068-full.json --retention full`
- `uv run --no-sync python -m benchmarks.v1.runtime_cost_audit /tmp/openboost-counts-068-summary.json --retention summary`

The practical same-container retention comparison is recorded separately.
