# 2026-09-05: Independent data and classification references

## Context

Following the production reset, F0.2 needs train-only transformations and A2/A3
classification geometry before public foundation components can be checked.

## Decision or Result

Keep immutable column transforms and class schemas distinct from tree geometry.
Numeric cuts follow direct linear order statistics; categorical candidates use
equality. Binary loss uses signed margins to avoid cancellation. Softmax exposes
its exact Hessian separately from the diagonal upper bound used for tree fitting.

## Changes

- [Sprint 003](../v1-sprints/003-data-classification-reference.md) records scope,
  hand fixtures, commands, results and reflection.
- Add independent data/classification reference modules and40 tests; extend the
  subprocess check that blocks all production imports.
- Keep full row identity/binding, categorical growth, persistence and production
  components explicitly pending. No API or performance claim is added.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  95 passed, no skipped, local macOS CPU/Python3.12.12/NumPy2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- Independent checks: hand cuts/category gain, finite differences, integer-weight
  replication, class permutations, two-round leaves/raw and isolated imports.
- CUDA, external baseline quality and formal E-gates were not run.

## Failed Attempts

- Initial tests failed collection on the two absent reference modules, confirming
  that this slice required new implementations. No production workaround was used.
- Code review identified clipping smaller than float64 can represent below one;
  reject that configuration explicitly to prevent an infinite binary base.

## Risks and Follow-ups

- These are tiny NumPy oracles, not optimized backends or public data containers.
  Constructors describe fixtures; public schema construction/serialization needs
  full validation when implemented. Supported category tokens are homogeneous
  strings or integers; other token types are rejected.
- Continue ranking/quantile/vector reference work, then remaining F0.2 and F0.3.
  Full v1 use-case scope and all production/evaluation obligations remain intact.

## Commits

- This slice: `test: add independent data and classification references for v1`.
