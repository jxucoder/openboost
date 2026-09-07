# 2026-09-07: Reuse immutable vector statistic layouts

## Context

The completed A6 profile spends 8.836 cumulative seconds in 1299063 calls to
_vector_indices. The same field-name layout is resolved repeatedly during scoring.

## Decision or Result

Cache only immutable name-to-index metadata, bounded to 128 layouts. Return fresh
lists to preserve caller ownership. Keep validation, arithmetic and custom callback
paths unchanged; this isolates one measured source of repeated work.

## Changes

- Private tuple layout resolver with bounded memoization.
- No training arrays, candidates, run state or model state in the cache.
- Invalid non-string layouts use the uncached resolver.

## Verification

The initial repeated-resolution test fails before implementation. Fifteen focused
tests pass afterward, including mutation isolation, invalid leaves and exact model
bytes/predictions against the original resolver across three growers/projected
layouts. Full regression: 1058 passed, one Linux-only skip; lint passes.

## Failed Attempts

Initial lint catches import grouping in the new test; corrected before commit.

## Risks and Follow-ups

No full-fit speedup is claimed. Re-profile the same approved packet and frozen
configuration. Scalar leaf validation/allocation remains substantial measured work;
any further change needs its own conformance and paired evidence.

## Commits

- Bounded vector schema reuse; parent `17ab9de`.
