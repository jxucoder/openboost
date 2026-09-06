# 2026-09-06: Auxiliary quality reports and censoring support

## Context

Primary metrics alone left required calibration, class, support and uncertainty
reports absent. Survival scoring must not treat a censoring time as a death or
extrapolate an unavailable censoring distribution.

## Decision or Result

Implement prediction-space diagnostics with explicit unsupported/undefined states.
IPCW Brier uses the frozen training G and supported grid. Only the G values used
in the formula are required; longer follow-up can contribute survival past an
earlier grid point without G extrapolation. Harrell C is separately named and
never described as IPCW. Paired intervals remain descriptive for five folds.

## Changes

- [Auxiliary metrics](../benchmarks/v1/auxiliary.py), documented primary definitions
  and tie/support conventions in the [evaluation README](../benchmarks/v1/README.md).
- [Quality report](../benchmarks/v1/quality_report.py): hashed survival/structure
  support, complete fold scores, missing-auxiliary ledger and paired summaries.
- Ten metric tests plus a complete five-fold hashed-survival report counterexample.

## Verification

- `uv run --no-sync pytest tests/v1/test_auxiliary_metrics.py tests/v1/test_quality_artifacts.py -n 0 -q`:
  13 passed. Hand calculations cover weighted AUC ties, censor exclusions,
  event/censor G ties, invalid grid support and undefined concordance.
- Rehashed but invalid censoring support still fails; absence is recorded and
  cannot turn into E3 acceptance. Exact bootstrap leaves global RNG untouched.

## Failed Attempts

No scoring failure in final fixtures. An initial lint check caught a missing
explicit zip strictness argument; it was corrected before commit.

## Risks and Follow-ups

These are diagnostics, not an E3 completion signal. Complete matrix/input-provenance
binding remains necessary. Survival auxiliary weights are currently unit-only;
Harrell C is not censoring-adjusted concordance. Structure strata do not establish
formula identifiability or parameter stability. Agent and execution gates remain open.

## Commits

- This slice: `eval: report auxiliary quality and supported survival scores`.

A further structural-artifact test verifies five-fold integration and rejects
rehashed but permuted structural row IDs. The focused metric/artifact total is
14 passing tests. Final full-suite and lint/documentation verification are recorded
with this slice below.

Final verification: all 472 v1 tests passed; Ruff and strict MkDocs passed.
