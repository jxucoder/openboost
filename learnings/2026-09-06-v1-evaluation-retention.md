# 2026-09-06: Select bounded history in current evaluation trials

## Context

The public default correctly remains full retention after Sprint 068. The current
real-data worker therefore still stores full traces unless it opts in explicitly.
This must be resolved before the larger frozen search preflight.

## Decision or Result

Use summary retention in openboost_worker's main recipe path and independent
quantile path. Record diagnostic_retention in training metadata. Do not introduce
an algorithm/search parameter, alter frozen model configurations or change the
public default. Other worker families require a separate audit before expansion.

## Changes

- Explicit summary policy on current A1–A12 worker calls and reported metadata.
- Tests inspect actual returned round payloads, not just a passed keyword.
- Preserve direct full-recipe and fresh inference parity checks for all adapters.

## Verification

Initial squared, Normal and quantile tests fail on full payloads. Focused current
worker suite: **86 passed**; full CPU regression: **1013 passed**. Lint/docs pass. Existing direct tests
compare against full recipes and fresh persisted predictions. No new GPU or real
selected-quality experiment is claimed.

## Failed Attempts

No retention flag is accepted from arbitrary search configuration: existing schema
validation stays unchanged. Summary is explicitly a diagnostic worker policy.

## Risks and Follow-ups

Summary does not bound model, current/raw, temporary or encoded-data memory. Actual
full 300/1000-round, 16-configuration resource preflight still follows 070's access/
resource checks. Frequency/severity and other worker families need separate audit.

## Commits

- Current evaluation summary policy; parent `63700db`.

## Composition follow-through

The frequency/severity worker also inherited full traces. A new test inspected both
actual recipe results and failed before the fix. Both component calls now request
summary retention and record that policy separately in component metadata. Frozen
configuration validation and the public default remain unchanged. The parametric
worker fits global controls and does not retain boosting round traces.

Verification: **10 composition tests pass**, including direct full-recipe identities
and fresh inference parity; **1019 CPU tests pass**, and scoped lint passes. No real
composition search or memory improvement is inferred from this policy change.
Implementation parent: `26a6797`.
