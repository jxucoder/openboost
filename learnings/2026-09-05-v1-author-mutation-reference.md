# 2026-09-05: Exact expectile, penalized leaf and ordered acceptance references

## Context

The F0.2 acceptance ledger identified missing D1/D3 formulas and D4's exact
six-trial acceptance schedule. These are development task oracles, not E5 results.

## Decision or Result

Enumerate expectile stationary intervals and penalized-pinball breakpoints plus
stationary points. Keep penalty normalization explicit: scaling row mass without
scaling the penalty changes the optimum. Ordered updates consume accepted state.

## Changes

- [Sprint 008](../v1-sprints/008-author-mutation-reference.md) records exact
  mathematics, two-round fixtures, verification and three-commit reflection.
- Add expectile objective/base, penalized residual leaf/tree and D4 ordered rule.
- 26 tests plus isolated production-import checks; update the acceptance ledger.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  255 passed, no skipped; local macOS/Python3.12.12/NumPy2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- Hand values, smooth-region finite differences, nonsmooth subgradient bounds,
  weight replication/scaling, routed two-round leaves and six-trial rejection.

## Failed Attempts

- Tests initially failed collection because the reference module did not exist.
  The first implemented mathematical batch passed; no thresholds were relaxed.

## Risks and Follow-ups

- No public extension package, real author-cost comparison, persistence or GPU was
  tested. Full best-state and per-run RNG integration is not proved by local raw
  snapshots or checking that this deterministic rule leaves global RNG unchanged.
- Complete the remaining categorical/vector growth and finite integration probes,
  audit F0.2 and then freeze F0.3. Do not expand reference work indefinitely.

## Commits

- This slice: `test: add exact author mutation references for v1`.
