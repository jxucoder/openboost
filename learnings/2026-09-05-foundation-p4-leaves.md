# 2026-09-05: P4.3 leaf reduction and explicit rule

## Context

P4.2 established numeric split/routing. Continue the minimum path to the bounded
leaf example and level-wise builder, keeping the goal-review decision to test
independent CPU packages before polishing complete GPU integration.

## Decision or Result

Separate real-row reduction from leaf arithmetic. LeafStatistics/reduce_leaves
expose compact G/H/count/active arrays; leaf_values composes the reduction with
an explicitly supplied rule. NewtonLeafRule is the default L2 reference. No
new tree type or alternate split criterion is introduced.

## Changes

- CPU compiled row sum and CuPy RawKernel reduction derive all statistics from
  routed rows. G/H already include weights; physical counts include zero-weight
  rows. -1 IDs and inactive slots are ignored; at most 511 slots. Invalid
  arrays/IDs/curvature and reduction overflow fail explicitly.
- Rules declare devices and return same-device contiguous finite float32
  values. A shared ExecutionContext can carry the fit RNG/channel; standalone
  calls create a seed-scoped context. Compact G/H copies and output detachment
  prevent scratch lifetime/mutation from corrupting model state. G/H mutation
  is rejected, including device mutation. No sample-sized protection copy.
- Default L2 rule handles zero/zero as zero, nonzero/zero as failure. Empty,
  inactive and zero-statistic leaves must be zero. The default rule rejects L1.
- A bounded rule is defined independently in tests with public API/array calls.
  The actual CPU trainer uses it through a root builder; GPU tests compose two
  rounds directly. These are distinct scopes, not a claim of a GPU Booster.

## Verification

- Initial tests failed collection because leaf_values/reduce_leaves were absent.
- Direct sample sums check nonuniform/zero weights, ignored and empty slots.
  Two-round hand oracle: unbounded raw=3.36, clipped raw=1; second weighted
  gradients [0.8,-3.6] versus [-1.5,-10.5] for y=[2,4], weights=[1,3].
- Output dtype/shape/device/finiteness, inactive values, input mutation, scratch
  ownership, L1 rejection and zero-curvature/overflow failures are tested.
- Final local and real-device results are recorded below.

## Failed Attempts

None beyond initial red collection at implementation time.

## Risks and Follow-ups

- Real T4 validation pending at the implementation commit.
- Atomic float32 reduction order is not deterministic; scalar checks synchronize.
- Default leaf arithmetic and clipped variants use the same existing numeric
  L2 split criterion; clipped split optimality is not claimed.
- Next is P4.4 LevelWiseBuilder and independent CPU package usability. GPU
  trainer, end-to-end quality/cost and external adoption remain later gates.

## Commits

- `383f5cd` — P4.2 evidence and earlier CPU extension usability checkpoint.

## Local verification

- `OPENBOOST_BACKEND=cpu NUMBA_NUM_THREADS=1 uv run --no-sync pytest
  tests/test_batch_leaves.py tests/test_batch_splits.py tests/test_batch_histograms.py
  tests/test_foundation_runner.py tests/test_experimental_objective.py
  tests/test_experimental_dispatch.py tests/test_experimental_persistence.py
  -n 0 -q`: **107 passed**.
- Production/changed-file lint and docs build passed; existing griffe warnings
  remain. GPU results must come from the committed wheel below.
