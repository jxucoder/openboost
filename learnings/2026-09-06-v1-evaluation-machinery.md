# 2026-09-06: Independent scoring and bounded evaluation workers

## Context

F0.3 needs independent quality decisions, leakage-resistant preprocessing, and
workers whose crashes/timeouts remain failures. See [Sprint 016](../v1-sprints/016-f0-3-completion.md).

## Decision or Result

Add separate layers for train-only encoding, raw prediction metrics, five-fold
comparisons, strict validation-only trial selection, hashed row-aligned artifacts,
and bounded process execution. Do not call these layers a completed E3 judge:
selection receipts and complete recipe/device coverage still require integration.

## Changes

- [Preprocessing](../benchmarks/v1/preprocessing.py) and actual five-split freezes;
  insurance claim and aggregate cases fit their own training populations.
- [Quality](../benchmarks/v1/quality.py), [artifact comparisons](../benchmarks/v1/quality_report.py),
  and [process execution](../benchmarks/v1/process_runner.py).
- [Search design](../benchmarks/v1/search-design.json): 16 configurations per listed
  comparator family, explicit CPU/GPU/time/retry limits; not a completed run manifest.

## Verification

- `uv run --no-sync pytest tests/v1/test_quality_evaluation.py tests/v1/test_quality_artifacts.py tests/v1/test_process_runner.py tests/v1/test_evaluation_preprocessing.py -n 0 -q`:
  17 passed. Tests cover failed/missing folds, per-target errors, NLL/censoring,
  invalid probabilities, row permutation after rehashing, timeout/nonzero exit,
  missing artifacts, unseen categories, and event/censor ties.
- Actual preprocessing freeze and replay passed using the pinned CPU environment.
- Current full suite: 422 passed, no skips. Ruff and strict MkDocs build passed.
  macOS/Python 3.12.12. No real model-quality or production-device claim.

## Failed Attempts

- Initial imports failed before quality and artifact modules existed.
- The live comparator probe later demonstrated that native GPU aborts cannot be
  caught as Python exceptions; independent process boundaries are mandatory.

## Risks and Follow-ups

- The process runner enforces wall time/threads, not a process-tree memory cap.
  Container limits and their provenance must be wired into full task execution.
- Quality comparison output intentionally leaves E3 false. A13 selection receipts,
  full expected matrix, frozen agent cohort and unseen H1/H2 remain required.
- AFT primary censored NLL is implemented. Independent test-support-aware IPCW
  Brier/C-index integration and classification auxiliary reports remain pending.

## Commits

- This slice: `eval: add independent scoring preprocessing and bounded workers`.
