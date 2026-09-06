# 2026-09-06: Multi-output regression closes a concrete CPU coverage gap

## Context

Sprint 035 found vector tree primitives but no complete A6 regression recipe.
Parent 83f6a9a; verification used this slice's dirty tree.

## Decision or Result

Compose independent scalar or shared vector trees with one joint transaction.
Expose split projections without reducing leaf/output dimension. Fit target
scaling on training weights only, preserve constant flags and persist inverse
scaling for original-unit predictions. Reject censored target semantics even
when all bounds happen to be finite.

## Changes

- objectives/recipes: MultiSquared, multi_squared and per-output MSE traces.
- multioutput.py: owned training-only scaler and original-unit inference artifact.
- Tests/docs: independent multi-round policies, projections, K=1, permutation,
  scaling, rejection and mixed fresh-process inference.

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- uv run --no-sync pytest tests/v1/test_public_multioutput.py -n 0 -q: 14 passed.
- uv run --no-sync pytest tests/ -m "not gpu and not benchmark" -q --tb=short:
  730 passed, macOS/Python 3.12.12/NumPy 2.3.5.
- uv run --no-sync ruff check src/openboost tests/v1/test_public_multioutput.py:
  passed.
- uv run --no-sync mkdocs build --strict; uv build --offline: passed.
- uv venv --python .venv/bin/python /tmp/openboost-multioutput-wheel-001;
  uv pip install --offline --python /tmp/openboost-multioutput-wheel-001/bin/python
  dist/openboost-1.0.0.dev0-py3-none-any.whl numpy==2.3.5: passed.
- Isolated Python -I from /tmp verified installed imports and executed every
  docs/v1/*.md Python block in fresh namespaces: eighteen passed.
  Build hash: [Sprint 036](../v1-sprints/036-a6-multioutput.md).

## Failed Attempts

Initial MultiSquared import failed before implementation. Constant-target test
then exposed floating weighted-mean error in variance-based constant detection.
Direct equality over positive-weight rows fixed the root cause. One unused
test import was removed by lint.

## Risks and Follow-ups

Raw recipe does not implicitly standardize targets. Callers preserve positional
output meaning and apply the saved scaler. Real subject-split A6 evaluation,
per-target comparisons, CUDA and performance remain unverified. Next follow
Sprint 035 item 2: prepared reuse and independent stopping/M32, not a GPU jump.

## Commits

Committed with this cohesive A6 slice; parent 83f6a9a.
