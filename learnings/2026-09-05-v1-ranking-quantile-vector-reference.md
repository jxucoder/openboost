# 2026-09-05: Distinct split and leaf mathematics for ranking, quantiles and vectors

## Context

F0.2 needs A4/A5/A6 counterexamples to test whether the foundation boundaries can
express query-local geometry, residual-based leaf solvers and vector payloads.

## Decision or Result

Keep pair aggregation, split statistics and leaf solving separate. A correct
quantile leaf does not imply the pseudo-gradient split objective will select a
split; this is a real quality limitation to evaluate, not a reason to force a tree.

## Changes

- [Sprint 004](../v1-sprints/004-ranking-quantile-vector-reference.md) records plans,
  exact normalization/sketch semantics, verification and reflection.
- Independent all-pairs ranking, frozen lambda weights/NDCG, pinball/residual
  quantile trees, shared vector stumps, split projection and train-only target scaling.
- 27 new tests plus extended production-import isolation. No production API added.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  122 passed, no skipped; macOS CPU/Python3.12.12/NumPy2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- Hand leaves, finite differences for pair/frozen-lambda geometry, query shifts,
  quantile nonsmooth optimality, two-round updates and output permutation/replication.
- No CUDA, real-data quality, persistence, sampling or formal E-gate was exercised.

## Failed Attempts

- Initial missing-module collection failure preceded implementation.
- Initial quantile two-round fixture assumed a profitable pseudo split despite
  weighted ties at its median. Keep this no-split counterexample; use another
  weight distribution with positive split gain for the residual-update test.
  The detailed derivation and fixtures are in Sprint 004.

## Risks and Follow-ups

- Vector coverage is a shared stump probe, not a full vector growth implementation.
  Split projection uses an explicitly declared diagonal sketch; full leaf outputs
  remain intact. Ranking sampling and query metric aggregation artifacts are pending.
- Complete positive/count and survival/AFT references, Normal/Formula, identity,
  state/run and remaining growth checks before F0.3/F1. All use cases remain required.

## Commits

- This slice: `test: add ranking quantile and vector leaf references for v1`.
