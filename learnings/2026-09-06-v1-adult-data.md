# 2026-09-06: Adult official test and typed raw records

## Context

Evaluation preparation remains the priority. A2 requires an unchanged official test
set, stratified splits within official training, missing categories and unit weights.

## Decision or Result

Pin the UCI archive, member bytes, typed parsed records and source-qualified row IDs.
Exclude fnlwgt, preserve categorical None, and use a fixed two-class label order.
The test set stays identical across all five seeds. Matching predictors across
sources are recorded without assuming they identify the same person.

## Changes

- [Sprint 014](../v1-sprints/014-adult-data-freeze.md) records scope and reflection.
- [Adapter](../benchmarks/v1/adult.py) and [freeze](../benchmarks/v1/datasets/adult.json)
  retain 32,561 training and 16,281 test rows. No fitted encoding or model is included.

## Verification

- Real archive `python -m benchmarks.v1.adult /tmp/openboost-v1-adult.zip --verify benchmarks/v1/datasets/adult.json`: matched.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`: 396 passed, no skips.
- Ruff over production, benchmarks/v1 and tests/v1: pass.
- Strict MkDocs build: pass. macOS/Python3.12.12/NumPy2.3.5.

## Failed Attempts

- Preimplementation collection failed on the missing adapter, as expected.

## Risks and Follow-ups

- Category encoding, baseline capability smoke, resource budgets and quality
  evaluators remain pending. No quality or GPU result is claimed.
- English-only repository prose is now an explicit user requirement.

## Commits

- This slice: `data: freeze Adult official test and stratified splits` (parent `f74df58`).
