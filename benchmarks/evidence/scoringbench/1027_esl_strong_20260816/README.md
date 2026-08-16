# 1027_ESL strong-baseline diagnostic

This is a clean, ScoringBench-shaped five-fold diagnostic shard. It
compares OpenBoost CPU against NGBoost, native XGBoost multi-quantile,
Gaussian XGBoostLSS, and CatBoost MultiQuantile. All 25 expected
dataset/model/fold rows completed; the OpenBoost and ScoringBench checkouts were
clean. The exact model constructors, package versions, platform, source SHAs,
and CI identity are in `openboost_manifest.json`.

Lower is better for every displayed score except raw coverage, whose target is
0.90. `Coverage error` is the fold-level mean of `abs(coverage_90 - 0.90)`.
The CRPS values are ScoringBench's histogram-based scores after every model is
converted to its common `DistributionPrediction` representation.

| Model | CRPS | RMSE | 90% coverage | Coverage error | 90% interval score | PIT KS | Train seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenBoost CPU | **0.3005** | 0.5459 | **0.8547** | **0.0510** | **2.4361** | **0.0894** | 1.3845 |
| XGBoostLSS | 0.3043 | **0.5384** | 0.7952 | 0.1048 | 2.8794 | 0.0991 | **0.1519** |
| NGBoost | 0.3077 | 0.5550 | 0.8137 | 0.0863 | 2.7371 | 0.1055 | 2.4431 |
| CatBoost quantile | 0.3249 | 0.5901 | 0.7172 | 0.1828 | 4.0657 | 0.1466 | 2.4481 |
| XGBoost quantile | 0.3257 | 0.6178 | 0.6127 | 0.2873 | 3.9807 | 0.2204 | 0.3869 |

On this shard, OpenBoost has the best mean CRPS, interval score, coverage
error, and PIT KS statistic. Relative to native XGBoost quantile, its CRPS is
7.7% lower, interval score is 38.8% lower, coverage error is 82.2% lower, and
RMSE is 11.6% lower. It wins four of five folds on CRPS and all five folds on
RMSE, interval score, coverage error, and PIT KS.

This is not an overall win. XGBoostLSS has 1.4% better RMSE and is about 9.1
times faster in these fit-only timings; OpenBoost has 1.2% better CRPS and
substantially better interval calibration. The timing is a cold/warm mixture
from one persistent process, is not a scale claim, and differed materially
across otherwise equivalent Actions runs.

The raw artifact also contains ScoringBench's reconstructed-density log score
and CRLS, but they are intentionally excluded from the comparison table.
Out-of-support targets are clamped to a quantile model's boundary density, and
CRLS is integrated over model-specific support; the upstream implementation
therefore does not make those two metrics comparable across these parametric
and finite-quantile predictions. Gaussian analytic NLL requires a separate
audit, and any cross-model density score requires a common support/tail rule.

The result is one small dataset, one seed, five correlated folds, unequal
model-specific default budgets, CPU only, and no paired confidence interval.
It is a diagnostic signal, not a library-level marketing claim. The next
quality decision must come from multiple untouched datasets and ultimately the
complete official suite. Hyperparameter changes prompted by this shard must be
developed elsewhere and not re-labelled as held-out evidence.

Source artifact: [GitHub Actions run 31925701435](https://github.com/jxucoder/openboost/actions/runs/31925701435), artifact `9257853524`, digest
`sha256:4044cc803958036d16c55aefed98c3142486e7ddba4bdfca61f364d5e7310765`.
`summary.json` hashes every frozen raw/result/provenance input except this
README and the summary itself, and records the unrounded diagnostic means.
