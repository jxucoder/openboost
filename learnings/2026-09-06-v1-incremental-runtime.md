# 2026-09-06: Evaluate proposed terms without accepted-ensemble replay

## Context

Sprint 066 attributes most instrumented CPU time to repeated prediction and
encoding. Sprint 067 first reproduced quadratic calls in a focused test: squared
4/8 rounds used 60/216 calls, Normal 120/432.

## Decision or Result

A proposal owns an internally derived, read-only candidate evaluation. Public
`preview_raw` exposes it to algorithm code. Acceptance carries these values forward;
rejection returns the original state. Keep independent full-model replay and exact
individual term additions. Public constructors and dataclass replacement cannot
inject prediction caches. Reuse immutable BinnedData by feature-data/binning
identity in accepted state, with copy-on-proposal encoding maps.

## Changes

- Runtime/recipes: evaluate new terms once per train/validation input and memoize
  only derived proposal state. Keep parent validation on every access.
- Tree: optionally accept identity-verified fitted encoding, including mixed data.
- Regression: linear calls, independent full replay, rounding-sensitive joint
  updates, failed scoring/rejection, read-only ownership, direct/replaced proposals,
  rebuilt states, stale parents and changed encoding identities.
- Preserve baseline artifacts; update operation-count and profile wrappers for
  the optional encoded tree call. Document the external `preview_raw` operation.

## Verification

Full CPU regression: **957 passed**. Ruff passes. Documentation build and offline
package build run before commit. Existing suite includes ordered/joint recipes,
validation stopping, offsets, heterogeneous runs and fresh specialized inference.
The count test now observes 8/16 squared and 16/32 Normal calls at 4/8 rounds,
with at most three encoding operations including learner preparation.

## Failed Attempts

The initial count test fails as expected on the old runtime. One new ownership
assertion initially expected TypeError where dataclass replacement correctly raises
ValueError for an init=False field; correct the test, preserve the rejection.

## Risks and Follow-ups

Private internal construction carries only values evaluated from validated immutable
state and proposals; it is not a public arbitrary-cache interface. Array metadata
must remain unmodified, as in the existing ownership contract. Model envelope checks
and identity hashing can still scale with term count, and full trace retention is
unchanged. Linear tree work is not a total-fit complexity or speed claim.

Next commit clean-source count evidence and rerun the exact frozen Sprint 066 CPU
protocol with resource checks. Preserve missing baseline cases, infrastructure
interruptions and partial profiles. No CUDA implementation or formal gate claim.

## Commits

- Incremental transaction/encoding implementation; parent `b9cc57a`.

### Clean-revision conformance evidence

At `e99e89c`, synthetic 4/8/16/32-round checks pass 2*K*T tree calls with exact
full replay. The installed D1–D4/custom-stopping/ordered M=1/8/32 suite passes,
including ten exact fresh-inference models after training packages are removed.
Source and artifact hashes match the clean revision. Evidence is in
[incremental-067](../benchmarks/v1/evidence/incremental-067/README.md).
These strengthen semantic and installed-boundary checks, not formal author gates.
