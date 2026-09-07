# 2026-09-06: Preserve claim-level units in the current Gamma worker

## Context

Parent 606487c. The frozen A8 exporter retains positive joined claim records,
policy-grouped splits and unit claim weights. The current worker lacked A8
support despite an existing public Gamma recipe.

## Decision or Result

Use the scalar Gamma recipe and original weights, retaining a log-mean model
and explicit positive-claim output. Do not introduce exposure or average claims
by policy. The selection score is the Gamma objective y/mean+log(mean), not a
fitted-dispersion likelihood or deviance statistic.

## Changes

- Current worker, inference and smoke harness add A8 without core changes.
- Weighted direct parity covers final/best selection and fresh persistence.
- Nonpositive targets, exposure and extra offsets fail explicitly.
- [Sprint 055](../v1-sprints/055-severity-worker.md).

## Verification

- Both new direct A8 cases failed on unsupported application before implementation.
- Current worker suite: 53 passed, including independent selected objective checks.
- Full CPU regression: 872 passed. Ruff, strict MkDocs and whitespace passed.
- All five frozen folds pass with exact fresh replay; independent Gamma objectives
  agree at rtol=1e-12/atol=1e-14. Source/output hashes match the
  [raw evidence](../benchmarks/v1/evidence/severity-055/README.md).
- Commands use UV_CACHE_DIR=/tmp/openboost-research-uv-cache and uv run --no-sync;
  macOS/Python 3.12.12/NumPy 2.3.5.

## Failed Attempts

The missing adapter was reproduced without changing the foundation or frozen
population. No averaging or exposure weighting approximation was introduced.

## Risks and Follow-ups

The next required row is A9, with distinct annualized target/exposure-weight
semantics and composition requirements. Real searches, quality, D5 and GPU
remain open. Short integration runs do not establish calibrated severity models.

## Commits

- This A8 worker slice; parent 606487c. Local only, no push.
