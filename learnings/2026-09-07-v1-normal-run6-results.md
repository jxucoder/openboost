# 2026-09-07: First Normal CUDA execution and acceptance-boundary failures

## Context

The user approved the 67-file upload and one T4 invocation frozen at `a5de692`.
Authorization commit `4143d18` was clean at dispatch. This is the sixth device
allowance; it has now been consumed. No retry or extra upload occurred.

## Decision or Result

The [raw run](../benchmarks/v1/evidence/cuda-normal-090/README.md) completed all
383 cases: **381 passed, two failed, no skips/errors**. The frozen verdict is
false and remains false. All 212 old tests, all twenty installed/fresh inference
checks, 23 Normal operation checks and 31 Normal recipe checks pass. Two of 96
mapped-runtime cases fail at exact backtracking-decision agreement.

The two failures are ordinary, ordered forward/reverse, depth-zero conflict
fixtures. GPU coefficients stop at `(8,4)` while the original-row reference
requires `(8,4,2,1,.5,.25)`. The reference starts at a stationary constant Normal
model with gradients at float64 rounding scale. This motivates a numerical
acceptance-policy investigation; retained failing logs do not establish its exact
device cause. Do not relabel it a pass or widen tolerances after measurement.

## Changes

- Archive all 79 raw artifacts and the unmodified manifest.
- Mark the live run-6 protocol consumed; preserve the original approved protocol
  inside the manifest and all frozen case/source hashes.
- Add archival tests for exact failure retention, full artifact integrity and
  all nineteen saved-model CPU replays.

## Verification

- Every recorded snapshot SHA256 matches `git show 4143d188b9635749308ef52382a1562928ae2612:<path>`.
  All 79 raw artifact hashes match. Recomputed `judge_run` equals the stored verdict.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest
  tests/v1/test_cuda_normal_manifest.py -n 0 -q` — **12 passed**. This checks
  archival integrity and CPU inference, not another device execution.
- Ruff passes for the updated manifest tests. Raw sources/tests/tolerances and
  earlier evidence are unchanged; staged diff inspected before archival commit.
- The generic ignore rule excludes `pytest.log`; force-stage this exact raw file.
  JUnit contains failure-rendering trailing whitespace. Preserve those bytes and
  verify every staged artifact against the manifest instead of editing raw data
  to satisfy a whitespace check. The non-artifact staged diff passes that check.

## Failed Attempts

One GPU invocation, two exact acceptance failures. All infrastructure/source/
installed-version checks passed; these are not image-build failures or skips.
The separate near-tie diagnostic still shows a zero-weight row prediction
difference, so its passing diagnostic cannot be counted as a structural repair.

## Risks and Follow-ups

Complete the retrospective from retained artifacts and reference mathematics,
then stop at the planned reflection boundary. Further GPU measurements require
a new frozen package/allowance. Normal phase exit, original P7, E4, remaining
CUDA recipes and independent-author accounting remain open.

## Commits

- `a5de692` — frozen run-6 package.
- `4143d18` — explicit upload and compute approval at dispatch.
