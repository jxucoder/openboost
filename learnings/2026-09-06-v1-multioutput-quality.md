# 2026-09-06: Report standardized A6 quality without hiding target failures

## Context

Parent 8d8272e. Selection now binds normalization to training data, but paired
quality reports still omitted the standardized-average metric required by A6.

## Decision or Result

Require hashed training row IDs, targets and scale for A6 quality cells. Recompute
population scaling and reject row overlap/misalignment or inconsistent metadata.
Report mean standardized RMSE alongside every original-unit target RMSE. Apply
paired comparison thresholds to each metric; an average never overrides a failed
target. Keep E3_pass false because selection provenance/full coverage are external.

## Changes

- quality_report.py: verified A6 normalization support and standardized metric.
- Five artifact tests including a better average with a failing target, constant
  target scaling and rejected missing/forged/evaluation-fitted scale support.
- [Sprint 047](../v1-sprints/047-multioutput-quality.md) and benchmark documentation.

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync pytest tests/v1/test_quality_artifacts.py::test_a6_reports_standardized_average -q -o addopts=''`:
  after correcting a fixture helper NameError, failed before implementation with
  unsupported primary metrics. The completed focused artifact file passes.
- `uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`:
  832 passed. Changed-file Ruff and strict MkDocs pass.
- CPU macOS/Python 3.12.12/NumPy 2.3.5. No real-data, GPU or timing evaluation.

## Failed Attempts

Initial fixture helper was scoped incorrectly and fixed before reproducing the
actual report gap. Lint also caught an unbound closure loop variable; the fixture
binds its fold explicitly. Neither was a foundation correctness finding.

## Risks and Follow-ups

Hashed supplied training data still requires trusted source/protocol provenance.
The report does not certify model selection, OS label isolation or complete v1
coverage. Actual paired A6 quality and other application adapters/searches remain
open. Sprint 017's normalization implementation gap is closed, not its full
F0.3 ledger. D5, CUDA and external author/adoption work remain. Nothing pushed.

## Commits

- This A6 quality-report slice; parent 8d8272e.
