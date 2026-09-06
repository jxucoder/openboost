# 2026-09-06: Formula and heterogeneous runs reuse the foundation

## Context

B06 tests whether coupled structured geometry and multiple independent algorithms
can reuse the scalar operations and state established by squared/Normal recipes.
This is also the reflection point after three recipe implementation commits.

## Decision or Result

Owned named structure binds row-aligned auxiliary inputs without becoming split
features. Saturation Formula exposes a full rank-one GGN; a damped SPD solve
produces unweighted directions, fitted through the same least-squares adapter,
trees, mapped terms and transactions as Normal. No diagonal fallback is hidden.
Sequential RunSpec execution shares immutable feature inputs while keeping
contexts, target/raw widths, round budgets, best states and error records separate.

## Changes

- data: owned structural role map and identity; unused structure is explicitly
  rejected by squared/Normal instead of silently ignored.
- objectives/recipes: saturation Formula prediction/base/geometry, damped full
  direction solve, joint updates and intermediate evidence.
- runs: immutable scalar options, unique-ID validation, sequential outcome/error
  records, returned-state identity checks and continuation after ordinary errors.
- Public Formula/run-many usage and capability boundaries documented.

## Verification

- Initial structural-role test failed before implementation with unsupported keyword.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run
  --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  581 passed, including nine new Formula/run tests. Python 3.12.12,
  NumPy 2.3.5, macOS. No CUDA execution.
- Three Formula rounds agree with independent explicit-inverse/exhaustive-growth
  reference gradients, full GGN, directions, coefficients, losses and raw output.
- Zero-damping rank deficiency fails; offset geometry agrees with explicitly
  shifted reference raw. Structure changes problem identity without changing
  feature identity; source mutation and writable re-enabling cannot change roles.
- Fresh-process persisted raw ensemble reproduces Formula output with separately
  supplied inference structure on numeric/missing/unseen observations.
- M=1/2/8 mixed recipes with K=1/2 and 0/1/2 rounds match independent same-ID and
  reversed-order results, with disjoint raw caches. Injected failure is recorded
  and the subsequent valid run completes. Duplicate IDs fail before any work.
- Invalid roles, unused structure, option overrides, fused mode and foreign
  returned state are rejected; options are copied from caller mappings.
- Ruff, strict MkDocs and offline build pass. All six public examples pass under
  Python -I from an isolated installed wheel outside the checkout.
  Wheel SHA256: 4f04e5e793035b0e232566a42bca9580256b3aab25afa88896a02892033ac310.

## Failed Attempts

Initial lint required import sorting. No numerical fixture or independent oracle
needed adjustment. Singular Formula geometry was an expected counterexample,
handled by explicit damping rather than weakened acceptance or pseudoinversion.

## Risks and Follow-ups

This is one saturation formula and sequential execution, not arbitrary symbolic
programs, early-stopping callbacks, resource scheduling, process isolation or
fused train-many. Each recipe still fits its own binning; only NumericData is
shared in the initial scheduler. Raw artifacts require explicit output transform
and structural inputs at inference. Full metrics here are tiny dense two-parameter
matrices, not a high-dimensional memory guarantee. Arbitrary recipe callbacks
must respect input ownership; execution is not a sandbox.

The three recipes support the geometry/statistics/state boundary but do not prove
that agents make changes faster or that all required applications work. Continue
B07 growth policies/categories, then vector and specialized leaf probes without
freezing these interfaces. Ordered Normal and all remaining application/evaluation
requirements remain open. F0.3 is incomplete. No quality/speed/GPU/adoption claim.

## Commits

- This slice: feat: add Formula geometry and sequential heterogeneous runs.
