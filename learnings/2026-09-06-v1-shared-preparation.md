# 2026-09-06: Explicit preparation reuse preserves run independence

## Context

Sprint 035 found that every recipe refitted binning despite shared data.
Parent 752a2ab; verification used this slice's dirty tree.

## Decision or Result

PreparedData fits training codes once and binds data/config identity. Every
built-in recipe accepts it explicitly; RunSpec forwards a dedicated field.
Original targets, weights and run state remain independent. M1/8/32 results
match independent preparation under reversed/regrouped execution.

## Changes

- binning.py: owned PreparedData and validated prepare_training resolver.
- recipes.py: twelve built-in preparation inputs; no other algorithm change.
- runs.py: explicit prepared field, reserved option name and failure isolation.
- Tests/docs: no-refit proof, identity/config mismatch and independent outcomes.

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- tests/v1/test_public_preparation.py: five cases pass in final regression,
  including M1/8/32; Binning.fit is disabled during all shared executions.
- uv run --no-sync pytest tests/ -m "not gpu and not benchmark" -q --tb=short:
  735 passed, macOS/Python 3.12.12/NumPy 2.3.5.
- uv run --no-sync ruff check src/openboost tests/v1/test_public_preparation.py:
  passed after import ordering fixes.
- uv run --no-sync mkdocs build --strict; uv build --offline: passed.
- uv venv --python .venv/bin/python /tmp/openboost-preparation-wheel-001;
  uv pip install --offline --python /tmp/openboost-preparation-wheel-001/bin/python
  dist/openboost-1.0.0.dev0-py3-none-any.whl numpy==2.3.5: passed.
- Isolated Python -I from /tmp verified installed imports and executed every
  docs/v1/*.md Python block in fresh namespaces: nineteen passed.
  Build hash: [Sprint 037](../v1-sprints/037-shared-preparation.md).

## Failed Attempts

Initial PreparedData import failed before implementation. Lint corrected test
import ordering. A signature-only test was removed; runtime equivalence/no-refit
checks establish the meaningful behavior.

## Risks and Follow-ups

Training-code reuse is not inference caching or a measured speed result.
Different fixed budgets are not independent early stopping. Add validation
patience/stop state and M32 stop/failure isolation next. No GPU, fusion, real
selection or phase-exit claim.

## Commits

Committed with the shared-preparation slice; parent 752a2ab.
