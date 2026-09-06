# 2026-09-06: Validation patience is separate from model acceptance

## Context

Sprint 038 M1 identified that best-model retention and different fixed budgets
did not establish independent stopping. Parent 7a61744. This slice executes
[Sprint 039](../v1-sprints/039-independent-stopping.md) without changing phase gates.

## Decision or Result

Public immutable StopState owns the round budget, optional patience, threshold
reference and progress. Recipes observe current validation once after an outer
round, including full rejection. Search trials do not consume patience. Strict
best-model selection remains independent of the threshold used to reset patience.
RunSpec passes scalar policy options through its existing public contract.

Independent scans of fixed-budget scalar/Normal fits determine expected stopping
prefixes. M=1/8/32 shared-preparation executions preserve those prefixes, best
models and stop records under reversal, regrouping and retry, with failures retained.

## Changes

- stopping.py: public finite-score/budget/patience state and immutable observation.
- recipes.py: all twelve recipes observe stopping and return FitResult.stop.
- Tests: thirty new cases, including all recipes, rejection, nonfinite failures,
  independent metric-sequence oracles and heterogeneous shared-run equivalence.
- Public docs, sprint index and agent guidance: implemented scope and next work.

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_public_stopping.py -q -o addopts=''`:
  initial missing-module failure; 29 passed after correcting the Normal fixture.
  The added numerical failure-isolation case passes in the full regression.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  765 passed, including all thirty new tests.
- `uv run --no-sync ruff check src/openboost tests/v1/test_public_stopping.py`:
  passed. Formatted changed Python files; no unrelated production cleanup.
- `uv run --no-sync mkdocs build --strict`; `uv build --offline`: passed.
- `uv venv --python .venv/bin/python /tmp/openboost-stopping-wheel-001`;
  `uv pip install --offline --python /tmp/openboost-stopping-wheel-001/bin/python
  dist/openboost-1.0.0.dev0-py3-none-any.whl numpy==2.3.5`: passed.
- Isolated Python `-I` subprocesses from /tmp verified site-packages imports and
  ran all twenty docs/v1 Python examples in fresh namespaces: passed.
- Environment: macOS, Python 3.12.12, NumPy 2.3.5. No CUDA or other OS verification.

Tested the parent's working tree plus this slice. Local wheel SHA256:
efc7705eb2cde914e4f1445cb087904bee02552df369b270728d7f80f4f7ec01.

## Failed Attempts

- The first focused test could not import StopState before implementation.
- The all-recipe fixture incorrectly supplied Formula structure to Normal.
  Normal correctly rejected it; fixed the fixture without loosening validation.
- Ruff required explicit non-strict zip for adjacent outcome pairs and import
  ordering; corrected before final lint/regression.

## Risks and Follow-ups

Stopping is an in-memory record, not a resume checkpoint. Recipe candidate
numerical rejection semantics remain unchanged. Validation observation evaluates
the metric over existing raw predictions; no end-to-end performance claim follows.
Next: installed public D2/D3 extension trials, D4 ordered updates and M3 real-data
integration under Sprint 038. No GPU, external adoption or complete E-gate claim.

## Commits

- This implementation commit — independent validation stopping across CPU recipes.
- `7a61744` — parent goal/progress review and execution plan.
