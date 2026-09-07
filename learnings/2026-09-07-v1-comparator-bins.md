# 2026-09-07: Explicit finite numeric comparator bin budgets

## Context

The 400-job A6 CPU plan revealed that baseline_worker rejected explicit bins and
used native defaults. The frozen shared budget must have a declared translation.

## Decision or Result

Optional bins is a strict integer from 2 to 256. XGBoost/LightGBM receive max_bin=B;
CatBoost receives border_count=B-1 for finite numeric intervals. NGBoost rejects
explicit bins because its current weak learner uses exact trees. Jobs omitting
bins retain native defaults; unknown native aliases remain rejected. The worker
already rejects nonfinite inputs, and this mapping makes no missing-value or
native categorical binning claim. Equal budgets do not mean equal quantization.

Official parameter references:
- [XGBoost max_bin](https://xgboost.readthedocs.io/en/stable/python/python_api.html)
- [LightGBM max_bin](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [CatBoost border_count](https://catboost.ai/docs/en/concepts/parameter-tuning)

## Changes

- baseline_worker translates once before library import and forwards parameters.
- bin_budget_smoke verifies actual installed native parameters on three-column A6
  targets, including a constant target, and saves models for fresh-process replay.
- Search planning keeps dispatch blocked pending full-budget installed resource
  checks and the other protocol obligations. Earlier planning freezes remain intact.

## Verification

Eight invalid-bin tests failed before implementation, then pass; NGBoost rejection
is explicit. A development run passes all six installed synthetic fits (B=7/255,
three libraries), parameter inspection, stopping records and exact fresh replay.
Versions: XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10 in the existing CPU environment.
A clean committed evidence run follows this implementation commit.

## Failed Attempts

No runtime failure in the development smoke. Initial tests exposed invalid bins
reaching native import/parameter validation instead of the declared worker contract.

## Risks and Follow-ups

Eight-round synthetic fits do not qualify real 300/1000-round execution, Linux
resource enforcement or quality. Native quantizers remain different. Preserve the
old plan as historical and generate a new input-pinned plan after verification.
No new data upload or full-search launch is needed for this local slice.

## Commits

Implementation follows `cd69c80`; evidence will be committed separately.

Validation: 1115 CPU tests passed, one Linux-only skip; changed-file lint and
documentation build pass.
