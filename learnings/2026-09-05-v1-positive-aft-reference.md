# 2026-09-05: Exposure roles and stable censored AFT references

## Context

F0.2 A7–A10 need independent positive-target and censored-likelihood formulas,
including exposure semantics and a consistent paid-count/severity target.

## Decision or Result

Treat Poisson exposure as an offset and annualized Tweedie exposure as a weight.
Keep event density and censored probability distinct. For far-right Normal tails,
retain the inverse-Mills correction directly so curvature does not cancel.

## Changes

- [Sprint 005](../v1-sprints/005-positive-aft-reference.md) records the bounded plan,
  hand values, tail verification, remaining work and three-commit reflection.
- NumPy/stdlib Poisson/Gamma/Tweedie, explicit bases/outputs, policy join probe,
  log-normal event/right-censored AFT and output transforms; no production API.
- 61 tests and extended production-import isolation. No new dependencies.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  183 passed, no skipped, local macOS/Python3.12.12/NumPy2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- Finite differences, two-round independent leaves, weight replication, exposure,
  policy association and time-density Jacobian tests. Far tails agree with erfc
  or independent Gauss-Laguerre quadrature at the declared fixture points.

## Failed Attempts

- Initial test collection failed on absent modules before implementation.
- An expanded unit-change test tried comparing a heterogeneous loss/g/h tuple as
  one NumPy array. Compare its fields separately; this was a test-container error,
  not evidence that the likelihood failed the unit transform.

## Risks and Follow-ups

- These are small mathematical references, not real ETL, insurance models or
  clinical results. Complete two-stage training/persistence, IPCW and real-data
  quality remain pending. Left/interval censoring and truncation are unsupported.
- Numeric domains must be representable; no silent clipping promises universal
  extreme-range support. F1 production implementations still need conformance.
- Continue Normal/Formula, identity/state/run, remaining grow and F0.3 freeze;
  all A1–A13 remain individually required.

## Commits

- This slice: `test: add positive target and AFT references for v1`.
