# 2026-09-05: Normal and Formula directional updates

## Context

F0.2 A11/A12 need geometry that differs from scalar Newton leaf fitting, including
joint/ordered parameters and rejection without residual state.

## Decision or Result

Solve unweighted Fisher/GGN directions before weighted least-squares tree fitting.
Evaluate candidate updates with the original loss. Full GGN does not establish
identifiability; a single structure input admits distinct equivalent parameters.

## Changes

- [Sprint 006](../v1-sprints/006-normal-formula-reference.md) records the bounded
  plan, formulas, verification, reflection and remaining F0.2 obligations.
- Independent Normal/Fisher, Formula/Jacobian/GGN, explicit 2×2 direction solves,
  initialization, separate Normal evaluator and immutable candidate-update probe.
- 25 tests plus production import isolation; no public trainer or runtime API.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  208 passed, no skipped; local macOS/Python3.12.12/NumPy2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- Hand Fisher solve, finite differences/Jacobian, CRPS/NLL, all three Formula
  directions, two-round joint/ordered updates, weight replication, rejected trials
  and exact reconstruction from accepted terms.

## Failed Attempts

- Initial tests failed collection before the reference module existed.
- Initial lint exposed test formatting; fixed before final verification.
- Review identified positive softplus parameters underflowing to zero; explicitly
  reject that unsupported range, rather than silently changing formula support.

## Risks and Follow-ups

- No real quality, parameter recovery, persistence, CUDA or E-gate is established.
  Update records do not implement the full best-state/callback/runtime contract.
- Next: state/run and identity references plus remaining acceptance mapping.
  Complete other F0.2 gaps and F0.3 before F1 public implementation.

## Commits

- This slice: `test: add Normal and Formula directional update references for v1`.
