# 2026-09-06: Annualized aggregate targets use exposure weights

## Context

Parent d7622f4. A9's frozen exporter binds eligible annualized paid totals and
exposure weights, but the current worker lacked a direct aggregate path.

## Decision or Result

Use public Tweedie at fixed evaluation power 1.5, positive exposure weights and
no additional offset. Persist explicit annualized output and power metadata.
Reject missing/nonpositive weights, negative targets and extra exposure/offsets.
The worker consumes a verified packet; the exporter establishes the weight's
source meaning. An arbitrary standalone packet cannot prove its own provenance.

## Changes

- Current worker, inference and smoke harness support direct A9 aggregate means.
- Weighted direct parity, selected objective, final/best selection and fresh replay.
- Invalid weight/target/unit inputs and altered persisted power are rejected.
- [Sprint 056](../v1-sprints/056-aggregate-worker.md).

## Verification

- Both direct A9 cases failed on unsupported application before implementation.
- Full CPU regression: 881 passed; current tests include independent weighted
  objective recomputation and persisted power validation.
- Five frozen folds pass exact fresh replay, independent weighted objectives and
  exact exposure-weight checks; target/period conversions pass. Source/output
  hashes match [evidence](../benchmarks/v1/evidence/aggregate-056/README.md).
- Ruff, strict MkDocs and whitespace checks pass. Commands use
  UV_CACHE_DIR=/tmp/openboost-research-uv-cache and uv run --no-sync;
  macOS/Python 3.12.12/NumPy 2.3.5.

## Failed Attempts

Missing support was reproduced without changing the frozen input population.
No second log-exposure offset or count-likelihood substitution was used.

## Risks and Follow-ups

Frequency-severity composition still needs eligible paid-event counts matched to
payment totals. Raw A7 claim counts do not satisfy that requirement. Complete
A9 quality/search, other adapters, D5 and GPU remain open.

## Commits

- This direct A9 aggregate slice; parent d7622f4. Local only, no push.
