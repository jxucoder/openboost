# 197_cpu_act HistogramBoost V2 development result

This is consumed development/tuning evidence, not held-out confirmation or
leaderboard evidence. It evaluates the frozen HistogramBoost V2 candidate on
the preregistered `197_cpu_act` development dataset. The run used five folds,
seed 42, a 3000-row cap, clean OpenBoost source `39bdb63`, and clean
ScoringBench source `a938a667`.

V2 used 50 distribution bins, 100 shared-vector trees, learning rate 0.05,
depth 6, and PSD Gauss--Newton curvature scale 1. It trained against the exact
piecewise-uniform histogram CRPS, interpreted `base_smoothing=1` as a total
Dirichlet concentration, selected a temperature from
`[0.5, 0.7, 0.85, 1.0, 1.2]` on an inner 80/20 split of each outer training
fold, refit on the full outer training fold, and losslessly subdivided every
output bin into two equal-density bins for evaluation. The extra inner fit is
included in training time. Outer-test targets were not used for model or
temperature selection.

| Model | CRPS | RMSE | Official 90% coverage | Official 90% interval score | Sharpness | Mean fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| HistogramBoost V2 | **1.287204** | **2.551506** | **94.80%** | **11.281116** | 3.719920 | 161.41 |
| CatBoost quantile | 1.326236 | 2.582497 | 71.73% | 13.329529 | **1.556934** | 141.00 |
| XGBoost quantile | 1.620136 | 3.698760 | 67.47% | 17.979673 | 1.619687 | **14.80** |

HistogramBoost V2 passed the frozen development gate. Relative to the
once-per-dataset selected strong baseline, CatBoost, it lowered mean CRPS by
2.943%, won four of five folds, lowered RMSE by 1.200%, and lowered the
official 90% interval score by 15.367%. Its mean absolute nominal-90% coverage
error was 4.80 percentage points, within the frozen five-point guardrail.

Against the frozen native XGBoost quantile baseline, HistogramBoost V2 won all
five folds and lowered mean CRPS by 20.550%, RMSE by 31.017%, and the official
90% interval score by 37.256%. This is a single consumed development dataset,
so it does not establish an overall, full-suite, or SOTA win.

The calibration metrics need careful interpretation. ScoringBench extracts
intervals using whole bin edges, so its coverage and interval scores are
representation-sensitive. V2's two-way subdivision preserves the represented
density, physical CRPS, mean, and continuous variance, while reducing the
coarse-bin envelope error. The official numbers above are therefore pinned
protocol metrics, not standalone calibration claims. Sharpness is diagnostic
only and is not part of the acceptance gate.

Within this one four-core Actions run, V2 averaged 161.41 seconds per fold,
including the inner calibration fit. It was 1.145x as slow as CatBoost and
10.907x as slow as XGBoost. These are implementation diagnostics, not general
speed claims. They make vector-tree GPU acceleration a concrete next systems
target now that the quality hypothesis has a positive signal.

The frozen evaluator returned `development_pass=true`, but an independent
protocol audit found that this commit's workflow did not execute that evaluator
inside CI and that the confirmation path was not yet phase-bound. Therefore
the untouched `537_houses` confirmation dataset remains locked until the gate
and provenance fixes are committed. The evaluator's
`confirmation_dataset_win=true` field is phase-agnostic and must not be read as
a confirmation result for this development run.

Run: [31932958804](https://github.com/jxucoder/openboost/actions/runs/31932958804),
job `95130441713`, artifact `9260176320`, digest
`sha256:572672e818cf60a295826f7b057bad0269f5121f5bfc31a258188eea12234adc`.
The artifact contains 15/15 valid rows with no errors, missing rows,
duplicates, unexpected rows, or non-finite metrics. The compressed dataset
matched the frozen SHA-256
`d00fd6bac2eda0821a04ff10277663d211127745a19a474c0feee41a16914fdc`.

