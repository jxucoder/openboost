# Sprint 072: A1 regression and A11 distributional quality

Status: planned. Mapping: N4–N5 / B14 / A1, A11 / R1, R6 / E3.
Depends on: the verified [071](071-real-multioutput-selection.md) selection pipeline.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Judge ordinary regression and distributional boosting on selected real models.
First verify that the independent scorer detects a mismatched Normal scale/link,
score normalization or missing calibration output while accepting the correct fixture.

## Work

- Reuse all five frozen Housing folds and the full preregistered search/budget per
  method, with validation-selected configuration and comparator before test release.
- A1: squared regression against supported engine baselines, original-unit RMSE,
  numeric/missing/weight semantics and fresh-process prediction replay.
- A11: Normal ordinary/Fisher and declared step variants, appropriate NGBoost or
  distributional comparison paths, NLL, CRPS, calibration and coverage with width.
  Dynamic backtracking stays dynamic; report its work and failures.
- Preserve NLL constants/units, distribution links and paired fold predictions.
  Count the shared source once in the six-source minimum. Keep official ScoringBench
  separate if used; a Housing experiment is not an official ScoringBench result.

## Acceptance and reflection

A1 and A11 each satisfy their E3 selected-quality contract and fresh inference;
NLL differences use the nats gate, not ratios of potentially negative values.
Coverage cannot pass without width and proper-score reporting. Store every trial,
selected-model receipt, fold score and total cost, including failed attempts.

If either gate fails, retain that application failure separately. Reflect on
whether the same preparation/transactions support both algorithms and whether
cost or model quality is the limiting factor. No global engine-parity claim.

## Results

Not run. Existing five-fold validation adapters are not selected-model acceptance.
