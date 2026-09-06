# 2026-09-06: Measure the full Covertype CPU bottleneck

## Context

Merged main 078ca61 contains Sprint 049's five 90-second Covertype timeouts.
Work continues locally on codex/covertype-cpu-profile. A static hypothesis favored
repeated prediction-time binning, but no timing evidence supported that priority.

## Decision or Result

A Unix diagnostic wrapper retains cProfile output on a 60-second soft alarm,
with the existing process runner imposing a 90-second hard cap. Run the exact
full-data fold-zero job under one thread. It exits 124 as intended; no fit pass.
Histogram aggregation uses about 30 seconds and hash updates about 22 seconds
in the interrupted instrumented window. Candidate generation repeatedly hashes
the same immutable row array. Prediction transforms account for about 2.28 seconds
cumulative. Prioritize invariant candidate row hashing before cache redesign.

## Changes

- profile_worker.py and two tests for result preservation, deadline artifacts
  and restoration of the process alarm state. Unix diagnostic only.
- [Sprint 050](../v1-sprints/050-covertype-profile.md) and
  [raw statistics/stacks](../benchmarks/v1/evidence/profile-050/README.md).

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_profile_worker.py -q -o addopts=''`: 2 passed.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`: 846 passed.
- Changed-file Ruff and strict MkDocs pass. Source and raw artifact hashes match.
- Exact diagnostic worker command, job/input hashes and outer outcome are recorded
  in manifest.json. It uses Sprint 049 fold zero, 60-second soft/90-second hard
  limits, one thread, local macOS/Python 3.12.12/NumPy 2.3.5. No memory cap or CUDA.

## Failed Attempts

The incomplete profile is intentionally not counted as a successful fit. Stack
snapshots support the observed histogram/hash paths. The initial prediction-cache
priority hypothesis is not supported by this window. Do not add overlapping
cumulative timings or extrapolate an end-to-end speed ratio from this run.

## Risks and Follow-ups

cProfile adds overhead; a local regression run overlapped the beginning of the
window. No performance acceptance or competitor comparison is implied. Next hoist
the per-invocation row hash with exact conformance tests, then rerun the unchanged
bounded workload. Histogram layout/aggregation is a separate follow-up. Required
real quality, D5, CUDA and adoption remain open. No new push or PR is authorized
by the earlier one-off merge request; this slice remains local.

## Commits

- This profiling slice; parent 078ca61.
