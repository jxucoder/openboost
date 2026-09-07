# 2026-09-07: Resident scalar rounds and transaction ownership

## Context

The user continued after 087 verified all 88 primitive CUDA cases. All prior
hardware allowances are consumed. 078-C must connect those public operations to
actual resident training while preserving the 065/068 ownership contracts.

## Decision or Result

Keep accepted/proposal raw storage private. Public raw snapshots are independent
device copies; accepting a proposal cannot share releasable proposal workspace.
Snapshot supplied trees and retain shared immutable internal terms with explicit
reference accounting. CPU inference export stays separate from training execution.
Use existing StopState independently from acceptance/best selection and retain
keyed run RNG behavior. This is a public scalar runtime, not implicit CPU fallback.

## Changes

[088](../v1-sprints/088-resident-scalar-training.md) freezes the public contract,
construction sequence, original-row two-round fixtures and acceptance boundary.
Weighted/missing, D2 and opposing validation targets exercise depth 0/1/2 and
cohort minima without depending on device implementation.

## Verification

`uv run --no-sync pytest tests/v1/test_device_round_reference.py -o addopts= -q`:
37 passed. Exhaustive float64 original-row tree/round results agree with public
CPU gradients, growth, transactions, validation and best selection. Ruff passes
for both new Python files. Local checks do not count as GPU execution.

## Failed Attempts

The resident scalar runtime is absent at the starting revision.
An initial verifier incorrectly passed BinnedData into PreparedData, which owns
fitting from raw data. The verifier now composes public CPU operations directly
with the frozen cuts; production preparation was not changed to fit the test.

## Risks and Follow-ups

Explicit release, borrowed inputs, model sharing and failed allocations must not
invalidate accepted state. A frozen dataclass alone does not provide this guarantee.
Retained user snapshots have a real storage cost; default history must avoid O(T*N)
raw arrays. Device training and formal R/C/A/E acceptance remain unverified.

## Commits

Commit fixtures/design, operations, runtime/recipe and validation packages in
separate verified slices. No push or new remote upload/run without authorization.
