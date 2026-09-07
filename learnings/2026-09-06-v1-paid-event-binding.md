# 2026-09-06: Match composition frequency to eligible severity events

## Context

Parent a041f1b. Direct A9 aggregate integration passed, but the public composition
helper accepts declared aggregates and cannot prove payment eligibility or joins.
A7 raw claim counts must not substitute for matched positive-payment counts.

## Decision or Result

Independently reconstruct count/amount aggregates from retained claim rows and
require exact equality with frozen source arrays. Bind to hashed A9 packets and
training IDs; require exact exposure weights and annualized target agreement.
Keep source population/preprocessing unchanged and emit no test labels.

## Changes

- paid_event_data.py: checked source-to-composition binding and five-fold CLI.
- Eleven tests exercise semantic mismatches and wrong partition/unit inputs.
- [Sprint 057](../v1-sprints/057-paid-event-binding.md).

## Verification

- Focused tests: 11 passed. Full CPU regression: 892 passed.
- All five real bindings pass; output/source hashes match the retained
  [manifest](../benchmarks/v1/evidence/paid-events-057/README.md).
- Ruff, strict MkDocs and whitespace pass. Commands use uv run --no-sync and
  UV_CACHE_DIR=/tmp/openboost-research-uv-cache; macOS/Python 3.12.12/NumPy 2.3.5.

## Failed Attempts

Lint found an import ordering issue, corrected before closure. No source data,
rows, thresholds or eligibility criteria were changed to obtain passing results.

## Risks and Follow-ups

The verifier is structurally independent aggregation over retained claims, not
an independently implemented raw ARFF parser. Frozen hashes bind the upstream
reader. Hash checks are not OS isolation. Next fit and replay composition on all
five packets; joint selection, full A9 quality, other adapters, D5 and CUDA stay open.

## Commits

- This input-binding slice; parent a041f1b. Local only, no push.
