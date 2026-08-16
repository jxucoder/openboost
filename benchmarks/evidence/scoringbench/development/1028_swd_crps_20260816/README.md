# 1028_SWD Gaussian CRPS-objective diagnostic

This is development/tuning evidence, not held-out or leaderboard evidence. It
compares the frozen NLL-trained OpenBoost baseline in
`../1028_swd_lr_sweep_20260816/baseline_lr_001/` with one candidate that
changed only `training_objective` from `nll` to `crps`. Both used 500 rounds,
learning rate 0.01, depth 3, identical regularization, seed 42, five folds,
sample cap 3000, 99 quantiles, and pinned ScoringBench commit `a938a667`.

| OpenBoost objective | CRPS | RMSE | 90% coverage error | 90% interval score | PIT KS | Sharpness |
|---|---:|---:|---:|---:|---:|---:|
| NLL | 0.355075 | 0.626043 | 0.0310 | **2.559809** | 0.087349 | **0.580131** |
| Gaussian CRPS | **0.353996** | **0.625552** | **0.0280** | 2.660561 | **0.080873** | 0.595394 |

The CRPS objective improved mean CRPS by 0.30% and won four of five paired
folds. It also improved PIT KS by 7.41%, absolute 90% coverage error by 9.68%,
and RMSE by 0.08%. However, it widened the predictive distribution by 2.63%
and worsened 90% interval score by 3.94% on every fold. This is a real but small
primary-metric improvement with a material guardrail regression.

It does not meet the stated benchmark goal. Candidate CRPS remains 4.62% worse
than native XGBoost quantile, 1.35% worse than Gaussian XGBoostLSS, 1.67% worse
than NGBoost, and 5.24% worse than CatBoost quantile on the already-frozen
five-model baseline. Therefore the objective is retained as a mathematically
tested development capability, not selected as the final ScoringBench model.
The next diagnostic should tune scale without changing the mean model, using
training-fold-only calibration; test-fold scale selection would be leakage.

Run: [31927426636](https://github.com/jxucoder/openboost/actions/runs/31927426636),
artifact `9258329504`, digest
`sha256:02f3827ee1787a6559f841cd277407d7faa1f2af2dd8031caf39794d8173594e`.
The run completed 5/5 expected rows from clean OpenBoost source `f092613` and a
clean pinned ScoringBench checkout. `summary.json` records unrounded effects
and SHA-256 hashes for every copied file. Fit-time differences from separate
Actions processes are not treated as speed evidence; reconstructed log score
and CRLS remain excluded from the decision for the documented evaluator
comparability reasons.
