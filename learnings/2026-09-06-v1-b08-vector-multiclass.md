# 2026-09-06: Vector leaves separate split and output dimensions

## Context

B08 needs multiclass and vector learners without duplicating tree growth or
restricting output dimension to split-statistic dimension. Sprint 027 starts at
6e5d6b0; verification ran with this slice's uncommitted changes.

## Decision or Result

All three existing growers accept separate split/leaf RowFields from the same
problem. Vector Newton fields retain original weights once. Shared topology
predicts [N,L], and explicit [L,K] mappings integrate with existing state. The
multiclass recipe uses one vector tree per round with softmax diagonal upper
bounds, preserving atomic rejection and persisted class order.

## Changes

- Stats/ops/tree: vector fields, per-channel diagonal leaves, summed gain and
  full leaf statistics independent of projected split fields.
- Artifacts: validated matrix payloads in tree-v3, arbitrary output maps and
  conservative per-channel finite envelope; earlier tree versions rejected.
- Objective/recipe/outputs: multiclass geometry, uniform-logit initialization,
  joint vector updates and softmax probability/label inference.
- Public docs: runnable multiclass example and updated capability boundaries.

## Verification

Commands use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- uv run --no-sync pytest tests/v1/test_public_vector_multiclass.py -n 0 -q:
  nine passing cases, independent topology/leaf/prediction and softmax oracles.
- uv run --no-sync pytest tests/ -m "not gpu and not benchmark" --tb=short:
  630 passed, Python 3.12.12 / NumPy 2.3.5, macOS.
- uv run --no-sync ruff check src/openboost tests/v1/test_public_vector_multiclass.py:
  passed.
- uv run --no-sync mkdocs build --strict; uv build --offline: passed.
- uv venv --python .venv/bin/python /tmp/openboost-vector-wheel-001; uv pip
  install --offline --python /tmp/openboost-vector-wheel-001/bin/python
  dist/openboost-1.0.0.dev0-py3-none-any.whl: passed.
- Isolated Python (-I, cwd /tmp) verified imported package was in that environment,
  extracted every fenced Python block from docs/v1/*.md and executed each in a
  fresh namespace: ten passed.
- Build identity and reflection: [Sprint 027](../v1-sprints/027-b08-vector-multiclass.md).

## Failed Attempts

The initial vector_newton import failed before implementation, confirming the
missing public boundary. No algorithm-oracle mismatch remained in final checks.

## Risks and Follow-ups

No CUDA, real-data quality, performance or agent/adoption claim. The diagonal
bound is not the exact softmax Hessian. All declared classes must appear in
training. Full A6 workflows and B09 ranking/quantile/penalized leaf work remain.
F0.3 is not closed by this construction slice.

## Commits

This entry is committed with the cohesive B08 vector/multiclass implementation.
Parent: 6e5d6b0.
