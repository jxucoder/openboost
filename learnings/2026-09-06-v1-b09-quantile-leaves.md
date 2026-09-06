# 2026-09-06: Routed residual leaves preserve original weight mass

## Context

B09/A5/D3 require original residuals and weights at leaves; summed Newton fields
cannot provide them. Parent 55f3b91; verification used this slice's dirty tree.

## Decision or Result

Separate owned residual context from weighted additive fields, retain global
row IDs in routed views, and validate matching problem identity. Ordinary and
penalized quantile solvers share all three growth policies. Positive penalties
are solved with a monotone subgradient scan, independently checked against
breakpoint/interior enumeration in the D3 reference.

## Changes

- leaves.py: owned residual context/views and exact scalar weighted pinball solver.
- tree.py: paired routed callback/context, rejecting ambiguous additive solvers
  and foreign problems.
- objectives/recipes: quantile base, pseudo-statistics and fixed/backtracking
  rounds; split regularizer and anchored leaf penalty remain separate.
- Documentation/tests: public runnable example and explicit CPU-only boundaries.

## Verification

Commands use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- uv run --no-sync pytest tests/v1/test_public_quantile.py -n 0 -q: 13 passed.
- uv run --no-sync pytest tests/ -m "not gpu and not benchmark" -q --tb=short:
  655 passed, macOS/Python 3.12.12/NumPy 2.3.5.
- uv run --no-sync ruff check src/openboost tests/v1/test_public_quantile.py: passed.
- uv run --no-sync mkdocs build --strict; uv build --offline: passed.
- uv venv --python .venv/bin/python /tmp/openboost-quantile-wheel-001;
  uv pip install --offline --python /tmp/openboost-quantile-wheel-001/bin/python
  dist/openboost-1.0.0.dev0-py3-none-any.whl numpy==2.3.5: passed.
- Isolated Python -I from /tmp checked the installed module path and executed
  every docs/v1/*.md Python block in a fresh namespace: twelve passed.
  Build hash and detailed reflection: [Sprint 029](../v1-sprints/029-b09-quantile-leaves.md).

## Failed Attempts

Initial public ResidualContext import failed before implementation. Ruff found
two unused imports during test development; removed before final checks.

## Risks and Follow-ups

ResidualContext is specifically scalar; it does not establish the final linear
or arbitrary structured-leaf context. Tests show CPU correctness, not real A5
quality, calibrated/noncrossing quantiles, GPU speed or lower author effort.
B10 positive-target/AFT construction is next; B09 pair-weight/sampling limitations
and broader workflow/evaluation gates remain open.

## Commits

Committed with this cohesive B09 routed quantile slice; parent 55f3b91.
