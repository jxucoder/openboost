# 2026-09-06: Exercise the current multiclass worker on full Covertype folds

## Context

Parent 0def101. Sprint 048 added A3 adapters and synthetic tests, but no current
full-dataset A3 result. The next slice executes the five frozen Covertype folds
without replacing the data with a subset or silently expanding worker budgets.

## Decision or Result

Use existing four-round, depth-two, 32-bin, seven-class, single-thread jobs with
90-second fit and 30-second fresh inference caps. Treat failures as evidence;
a full-fold validation smoke is not a full quality search or a speed comparison.

## Changes

- [Sprint 049](../v1-sprints/049-covertype-worker.md): bounded evaluation plan.
- [Raw evidence](../benchmarks/v1/evidence/covertype-049/README.md): five timeouts,
  complete jobs/input identities, worker logs and bounded execution records.

## Verification

Use UV_CACHE_DIR=/tmp/openboost-research-uv-cache.

- `uv run --no-sync python -m benchmarks.v1.openboost_worker_smoke /tmp/openboost-covertype-049 --applications A3`.
- Source parsing, all five frozen preprocessing records and packet identities are
  checked before fitting. The source contains 581,012 rows and 54 input features;
  original labels 1–7 are encoded as 0–6 by the pinned reader.
- Local macOS/Python 3.12.12/NumPy 2.3.5. No memory cap or CUDA.

## Failed Attempts

All five folds timed out at 90 seconds and produced no model. No predictions
were available for fresh replay, so no fold passes. Worker logs were empty.
The process records show group termination under the declared cap, not a
mathematical exception or an identified hotspot. Source/artifact hashes match.

## Risks and Follow-ups

Full model selection, quality/calibration, required comparator controls, D5 and
GPU/adoption gates remain open. Packet separation is not OS label isolation.
Next: profile the same full-data input with bounded phase/stack diagnostics,
separating preparation, initialization, tree construction, prediction and model
state transactions. Repeated prediction-time binning is a static hypothesis only.
Do not increase the budget or redesign caches without measurement. No code was
changed; latest CPU regression remains 844 from Sprint 048, not rerun here.
Strict MkDocs and git diff checks pass. Nothing pushed or published.

## Commits

- This Covertype evidence slice; parent 0def101.
