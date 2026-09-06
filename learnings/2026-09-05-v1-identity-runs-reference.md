# 2026-09-05: Prepared identity, isolated runs and comparable selection

## Context

F0.2 A13/D5 needs concrete run isolation, best-state and stable random-key evidence;
C1 requires content/row identity instead of object ID or shape.

## Decision or Result

Preserve prepared row IDs alongside the content digest so bind can reject a caller
that consistently misorders all roles. Run heterogeneity does not justify comparing
incomparable validation losses: select only within one problem identity.

## Changes

- [Sprint 007](../v1-sprints/007-identity-runs-reference.md) records the bounded
  implementation, fixed key, results and reflection.
- Typed hash/bind oracle, real tiny scalar/vector squared-tree runs, immutable best
  terms, independent stopping/failure and stable per-run sampling.
- [F0.2 acceptance ledger](../v1-sprints/f0-2-acceptance-ledger.md) maps all use cases
  and author tasks to evidence and remaining work. F0.2 is still open.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  229 passed, no skipped; local macOS/Python3.12.12/NumPy2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- Independent/sequential/reordered/regrouped records agree; M=1/8/32, K=1/2,
  failure/retry, best reconstruction, ID/data changes and isolated imports checked.

## Failed Attempts

- Initial tests failed collection before the module existed; formatting lint fixed.
- Review found that an opaque digest alone cannot verify claimed prepared row order.
  Add immutable row metadata and a counterexample where every role shares the wrong order.

## Risks and Follow-ups

- This is a sequential unit-weight squared-loss probe, not a production scheduler,
  reusable cache or batched/GPU implementation. No timing claim follows from regrouping.
- Full artifacts, runtime callbacks and other recipe integration remain pending.
- Next D1/D3/D4 exact fixtures, then remaining grow/integration references and F0.3.

## Commits

- This slice: `test: add identity and isolated run references for v1`.
