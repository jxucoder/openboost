# 2026-09-06: Persist the dependency roles of a two-model mean

## Context

A9 requires matched positive-payment frequency/severity composition with saved
dependencies, distinct from Tweedie fitting. Parent bf7ef2c; checks used the
slice's dirty tree.

## Decision or Result

Bind declared paid-count/total aggregates to frequency and count-weighted
severity-average problems. Embed both raw models in a role-specific artifact.
Require aligned inference policy IDs and explicit exposure/offsets, and expose
rate, count, severity, annualized and period means.

## Changes

- composition.py: aggregate validation/problem assembly and FrequencySeverity
  bundle with fixed output semantics and strict persistence.
- artifacts.py: public nested Model.from_record shares the existing loader checks.
- Tests/docs: matched weights, multiple rounds, policy alignment, output units
  and corrupted/fresh-process inference.

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- uv run --no-sync pytest tests/v1/test_public_composition.py -n 0 -q: 9 passed.
- uv run --no-sync pytest tests/ -m "not gpu and not benchmark" -q --tb=short:
  693 passed, macOS/Python 3.12.12/NumPy 2.3.5.
- uv run --no-sync ruff check src/openboost tests/v1/test_public_composition.py:
  passed.
- uv run --no-sync mkdocs build --strict; uv build --offline: passed.
- uv venv --python .venv/bin/python /tmp/openboost-composition-wheel-001;
  uv pip install --offline --python /tmp/openboost-composition-wheel-001/bin/python
  dist/openboost-1.0.0.dev0-py3-none-any.whl numpy==2.3.5: passed.
- Isolated Python -I from /tmp verified the installed path and executed all
  docs/v1/*.md Python blocks in fresh namespaces: sixteen passed.
  Build hash: [Sprint 033](../v1-sprints/033-b10-frequency-severity.md).

## Failed Attempts

Initial composition import failed before implementation. One unused test import
was removed by lint.

## Risks and Follow-ups

The helper validates supplied aggregates, not raw eligibility/joins or provenance.
It uses shared policy predictors and count-weighted positive averages; it does
not construct claim-specific covariates. Component validation selection is not
joint aggregate-quality selection. No real A9, calibration, GPU or speed claim.
AFT target/scale/output semantics are next; broader workflow gates remain open.

## Commits

Committed with the cohesive B10 frequency-severity slice; parent bf7ef2c.
