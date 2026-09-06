# 2026-09-06: Composable depthwise numeric trees

## Context

B04 had public scalar operations but no assembled production learner. The next
probe was whether a grower could use these operations and ordinary callbacks,
with inference state independent of training rows and an exhaustive oracle.

## Decision or Result

Depthwise growth composes public operations, preserving original routed positions.
A leaf cap selects highest-gain splits within a layer before allocating stable
child IDs. Scoring executes once per legal candidate; caching avoids a second
callback invocation changing the layer decision. The inference artifact owns its
transformer and explicit topology rather than relying on heap-index children.

## Changes

- `src/openboost/tree.py`: depthwise growth with scoring/legality/leaf callbacks;
  immutable numeric tree inference, graph validation and strict JSON persistence.
- `tests/v1/test_public_tree.py`: independent topology/prediction comparisons,
  callback changes, corruption rejection and fresh-process persistence.
- Public docs and Sprint 020 describe the exact implemented boundary.

## Verification

- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run
  --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  553 passed, including 26 tree cases. macOS, Python 3.12.12, NumPy 2.3.5.
- Exact topology and numerical leaf/prediction comparisons against independent
  exhaustive row growth at five depth/leaf budgets with weights and missing data.
- Replaced feature scoring, independent information feasibility and leaf solving
  change results; doubled custom leaves double learner output.
- Cycles, shared/unreachable nodes, invalid index/routing/leaf/schema/cuts,
  duplicate fields, nonfinite callbacks and empty-child admission are rejected.
- Ruff, strict MkDocs and offline sdist/wheel build passed. All three public docs
  examples ran under Python -I from an isolated installed wheel outside the repo.
  Wheel SHA256: 55308281657b883dd9cdaa80d8ac9b6f95c195c12c88706a1e6fa7f4673eb714.

## Failed Attempts

The shell has no bare `python`; the documentation update was rerun using the
project interpreter. Initial lint flagged ambiguous child variable names and a
loop callback closure; explicit child names and a bound cache resolved these.
No acceptance criteria or independent reference behavior changed.

## Risks and Follow-ups

Dense CPU operations have no measured speed advantage. The tree is a scalar
learner, without ensemble base/coefficient/offset or transaction integration.
No categories, specialized/vector leaves, CUDA or resume checkpoint is implemented.
Next: tree terms in immutable run state and complete squared/Normal recipes,
then B06 Formula and heterogeneous run probes before stabilizing interfaces.
F0.3 and all full v1 evaluations remain open; no required application was removed.

## Commits

- This slice: feat: assemble composable depthwise numeric trees.
