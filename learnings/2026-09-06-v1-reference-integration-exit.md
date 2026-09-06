# 2026-09-06: Reference composition and F0.2 exit

## Context

F0.2 still needed finite offset, two-stage, quantile ensemble and accepted/best
state compositions before moving to the frozen evaluation protocol.

## Decision or Result

Close independent reference preparation after auditing all required application
families and development tasks. This does not close production or evaluation gates.
Best restoration needs a fresh version to reject abandoned-future proposals while
retaining the matched terms, coefficients, train/validation caches and logical step.

## Changes

- [Sprint 010](../v1-sprints/010-reference-integration-exit.md): bounded plan,
  nine integration tests and phase reflection.
- [Acceptance ledger](../v1-sprints/f0-2-acceptance-ledger.md): required coverage
  and remaining work assigned to F0.3/F1–F5 without dropping use cases.
- Immutable finite ensembles and Normal state probes; production import blocker
  now exercises the positive ensemble composition too.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  288 passed, no skipped; macOS, Python 3.12.12, NumPy 2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict`: pass.
- Offset exactly once, paid-count join through model predictions, three quantiles,
  two-round ordered commits, atomic rejection and base/nonzero best restoration.

## Failed Attempts

- Nonzero-best fixture initially compared unlike metric definitions; initialization
  and all proposals must use the same validation score for meaningful selection.
- Review found broadcastable row/shape mismatches could evade numerical comparison;
  explicit checks now reject them before commit, with a regression case.

## Risks and Follow-ups

- No public runtime, persistence, CUDA, external quality or author-cost validation.
- Next: F0.3 datasets/hashes, installed-library capability smoke, budgets, held-out
  tasks and artifact judge. Do not turn synthetic reference success into parity claims.

## Commits

- This slice: `test: integrate reference models and close v1 F0.2` (parent `972f80a`).
