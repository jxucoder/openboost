# 197_cpu_act HistogramBoost preregistered development result

This is development/tuning evidence, not held-out, confirmation, or
leaderboard evidence. It is the first run of the frozen HistogramBoost
candidate on the preregistered `197_cpu_act` development dataset. The candidate
used 50 distribution bins, 100 shared-vector trees, learning rate 0.05, depth
6, and PSD Gauss--Newton curvature scale 1. The benchmark used five folds,
seed 42, a 3000-row cap, and pinned ScoringBench commit `a938a667`.

The strong baseline was selected once per dataset as the lower-mean-CRPS model
between native XGBoost quantile and CatBoost MultiQuantile. CatBoost was the
selected baseline.

| Model | CRPS | RMSE | 90% coverage | 90% interval score | Sharpness | Mean fit time (s) |
|---|---:|---:|---:|---:|---:|---:|
| CatBoost quantile | **1.326236** | **2.582497** | 71.73% | **13.329529** | **1.556934** | 144.43 |
| HistogramBoost | 1.354233 | 2.591390 | **98.23%** | 13.496925 | 6.476879 | 84.34 |
| XGBoost quantile | 1.620136 | 3.698760 | 67.47% | 17.979673 | 1.619687 | **13.59** |

HistogramBoost did **not** pass the frozen development gate. Its CRPS was
2.111% worse than CatBoost, just beyond the 2% non-inferiority threshold, and
it won only two of five folds rather than the required three. Its absolute
90% coverage error was 8.23 percentage points rather than at most five. The
interval-score and RMSE guardrails passed. Therefore the untouched
`537_houses` confirmation dataset remains locked.

There is a useful but limited positive signal: HistogramBoost beat the native
XGBoost quantile baseline on all five folds, lowering mean CRPS by 16.41%,
90% interval score by 24.93%, and RMSE by 29.94%. This is one development
dataset and cannot support an overall win claim.

The failure shape is informative. HistogramBoost's mean prediction RMSE was
within 0.34% of CatBoost, but its distribution was much wider: sharpness
6.48 versus 1.56 and 98.23% empirical coverage at the nominal 90% interval.
The next development work should diagnose why the finite-bin PMF retains too
much tail mass before changing model capacity. It should not use the untouched
confirmation dataset.

Within this single four-core Actions run, HistogramBoost fit in 84.34 seconds
per fold on average: 1.71x faster than CatBoost but 6.21x slower than XGBoost.
These timings are implementation diagnostics, not a general speed claim.

Run: [31930502694](https://github.com/jxucoder/openboost/actions/runs/31930502694),
artifact `9259424361`, digest
`sha256:9b8d5676727cbece7a61ac1067c6705089cb69b9e8cc3718bd0e79e34b056f6a`.
The artifact contains 15/15 valid rows with no errors, missing rows, duplicates,
or non-finite metrics. OpenBoost source `835335a` and ScoringBench were clean;
the compressed dataset matched the preregistered SHA-256
`d00fd6bac2eda0821a04ff10277663d211127745a19a474c0feee41a16914fdc`.
The fail-closed evaluator was committed as `3cbd763` before this result was
observed.
