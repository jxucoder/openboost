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

## Same-container full/summary pairs

[Manifest](practical/manifest.json) records clean `ae8c7af`, wheel/source/input hashes,
explicit sixteen-fit/two-profile amendment, actual-image resource checks, settings
and exact invocation. All sixteen fits and both summary profiles pass. Full and
summary run sequentially in the same container with the same two-thread pool,
8-GiB address ceiling and original per-worker deadlines. The input is the unchanged
frozen Housing protocol, with no test labels. Each summary retains all outer-round
records, zero per-round array bytes and exact full-model replay.

All eight pairs have **exact raw predictions and byte-identical model JSON**.
Independent metrics, actual loaded source hashes and all raw artifact hashes were
verified. [Comparison](practical/comparison.json) also retains end-to-end times.

| Case | Full fit s | Summary fit s | Full peak MiB | Summary peak MiB | Full arrays MiB | Summary arrays MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| squared-8192-4 | 0.334 | 0.332 | 84.871 | 85.129 | 0.5625 | 0 |
| normal-8192-4 | 0.664 | 0.651 | 87.727 | 86.676 | 2.1250 | 0 |
| squared-2048-32 | 2.393 | 2.381 | 85.270 | 85.422 | 1.0156 | 0 |
| normal-2048-32 | 4.826 | 4.819 | 88.660 | 85.855 | 4.0312 | 0 |
| squared-8192-32 | 2.691 | 2.732 | 90.371 | 85.684 | 4.0625 | 0 |
| normal-8192-32 | 5.477 | 5.440 | 103.535 | 87.277 | 16.1250 | 0 |
| squared-8192-128 | 12.990 | 12.937 | 103.320 | 88.500 | 16.0625 | 0 |
| normal-8192-128 | 26.564 | 26.480 | 156.887 | 93.441 | 64.1250 | 0 |

Summary scalar records, current arrays and models still occupy memory. Some small
cases have slightly higher summary RSS; do not claim uniform process savings.
At 128 rounds the observed RSS reduction is about 14.8 MiB squared and 63.4 MiB
Normal. These are single ordered observations, not repeated estimates or a formal
memory/cost gate. Guest RSS remains separate from the stricter address-space cap.

Both summary profiles complete with 256/512 tree evaluations for squared/Normal,
zero trace-array bytes and 128 retained round records. The original pstats text
retains its trailing blank line for hash integrity. Full/summary timing is similar
in this diagnostic; this change targets retained history rather than tree-building
work. No full-search quality, GPU or author-effort gate follows.

Reproduce by copying/rebuilding the original frozen input/protocol into a fresh
directory, then run:
`uv run --no-sync python -m benchmarks.v1.practical_cpu_profile run /tmp/openboost-retention-cpu-068 --paired-retention`.
The manifest contains exact per-child commands, software and resource details.
