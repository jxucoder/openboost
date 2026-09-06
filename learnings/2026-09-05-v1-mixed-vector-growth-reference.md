# 2026-09-05: Full mixed-feature and vector growth references

## Context

The F0.2 ledger still needed native categorical full trees, multilevel vector
payloads and training-transform-to-raw-prediction integration.

## Decision or Result

Bind fitted transforms to the reference tree and retain full vector leaves even
when split statistics are projected. A split that helps one output can have
negative total gain because splitting another output incurs extra regularization.

## Changes

- [Sprint 009](../v1-sprints/009-mixed-vector-growth-reference.md) records the
  bounded plan, failed assumption, tests, results and reflection.
- Independent exhaustive mixed-feature grow for depthwise/best-first/symmetric,
  immutable transform/tree records, full vector leaves and identity integration.
- 24 tests plus isolated production-import checks; original scalar/stump references
  remain separate comparisons.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q`:
  279 passed, no skipped; local macOS/Python3.12.12/NumPy2.3.5.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py`: pass.
- Multilevel two-round leaves, K=1 scalar comparison, three growth policies,
  categorical equality/unknown, row conservation, weight replication and raw transforms.

## Failed Attempts

- Initial collection failed on the absent module.
- Initial vector fixture assumed a second split helped the overall objective.
  Summed gain was actually -1 under lambda=1; retain the correct no-split case,
  and use a distinct positive-gain fixture to exercise full depth.

## Risks and Follow-ups

- Reference layout and immutable records are not production data/artifact APIs.
  No leaf budget, GPU, real quality or serialization claim is made here.
- Next finite offset/two-stage and best/RNG integration, then audit F0.2 and freeze
  F0.3. Full required scope remains unchanged.

## Commits

- This slice: `test: add mixed feature and full vector growth references for v1`.

Pre-commit review also added explicit finite checks for combined candidate/layer
scores: finite child scores can overflow when summed. The added counterexample
passes; the final suite count is 279.
