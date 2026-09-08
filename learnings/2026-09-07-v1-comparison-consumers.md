# 2026-09-07: Separate objective comparison consumers

## Context

The approved [092-C ownership plan](../v1-sprints/092-comparison-consumers.md)
requires three independently anchored consumers. Correct comparison mathematics
alone had not changed acceptance, best selection or patience.

## Decision or Result

CPU Normal now uses objective evidence for all three decisions. The run-7 stored
worsening candidates are rejected, while tiny genuine improvements lost in total
NLL rounding advance acceptance, best and patience. The current/best/patience
anchors follow the prescribed five-transition sequence independently. Reporting
scores retain their original values.

## Changes

- CPU resolve accepts explicit compare and replays an owned best validation anchor.
- StopState.observe_change consumes proved improvement against min_delta; the
  Normal recipe owns and replaces its immutable patience snapshot on that decision.
- Normal trial and stopping comparison records survive full and summary retention.
  TraceSummary explicitly permits LossChange, an immutable scalar record; arbitrary
  author records and arrays remain rejected.

## Verification

- Before implementation, all ten new consumer cases failed, including a real
  recipe that rejected equal-score improvements and stopped prematurely.
- Focused command: `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_comparison_consumers.py tests/v1/test_trace_retention.py tests/v1/test_public_normal.py tests/v1/test_public_stopping.py tests/v1/test_incremental_runtime.py -n 0 -q --tb=short` — 100 pass.
- Full CPU regression: `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short` — 1646 pass, one Linux-only skip.
- Ruff passes for production and the new consumer test. MkDocs builds with the
  existing execution-page link warning for evidence outside the documentation tree.
- No CUDA execution, upload, frozen-oracle change or tolerance adjustment.

## Failed Attempts

The first consumer implementation attempted to put LossChange into the existing
scalar-only summary validator. It correctly rejected the record. The summary
contract now names that single supported immutable scalar record explicitly;
generic dataclass traversal would risk retaining author arrays or state.

## Risks and Follow-ups

CPU best selection adds a full validation-model replay. This is a correctness
reference, not an optimization claim. Resident best ownership, device consumers,
383-case requirement bindings and the 092-D freeze remain open. All seven device
allowances are consumed; no hardware execution is authorized by this local slice.

## Commits

- This entry accompanies the CPU consumer implementation commit.
