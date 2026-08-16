# CRPS distribution experiment V2

This protocol freezes the second HistogramBoost candidate before its full
five-fold `197_cpu_act` run. The V1 five-fold artifact and V2 fold-0 diagnostics
are consumed development evidence. `537_houses` remains untouched and locked.

## Why V2 exists

V1 beat the frozen native XGBoost quantile baseline on all five folds, but was
2.111% worse than CatBoost on mean CRPS, won only two of five CatBoost folds,
and retained an over-wide PMF. Audits identified three correctable causes:

1. training used a midpoint ranked-probability approximation while evaluation
   used exact piecewise-uniform histogram CRPS;
2. additive smoothing was applied once per bin, so total prior strength grew
   with output resolution;
3. whole-bin interval extraction made the 50-bin representation coarser than
   the CatBoost representation.

V2 aligns the training score with evaluation, treats smoothing as total
Dirichlet concentration, selects probability temperature using only an inner
split of each outer training fold, and losslessly subdivides bins for evaluation.

## Frozen data and benchmark

- Development: `197_cpu_act` from the commit-pinned registry
  `crps_distribution_v1.json`.
- Untouched confirmation: `537_houses` from the same registry. The launcher
  must reject it unless `--allow-confirmation` is explicit.
- Dataset bytes and SHA-256 values are unchanged from V1.
- ScoringBench commit:
  `a938a667b7839b41e9272929010573410301c0b4`.
- Five folds, one repeat, seed 42, sample cap 3000, CPU execution.
- Baselines: native XGBoost quantile with 100 rounds and 50 quantiles; CatBoost
  MultiQuantile with 1000 rounds and 99 quantiles. Both use two threads.
- Exactly 15 finite, error-free result rows are required.
- CRPS is primary. Log score, CRLS, CDE, and DPD are excluded from decisions
  because support and density-grid differences make them unsuitable here.

## Frozen V2 candidate

The benchmark model name is `openboost_histogram_cpu_v2`:

```python
OpenBoostHistogramWrapper(
    n_distribution_bins=50,
    n_trees=100,
    learning_rate=0.05,
    max_depth=6,
    n_feature_bins=254,
    curvature_scale=1.0,
    temperature_grid=(0.5, 0.7, 0.85, 1.0, 1.2),
    calibration_fraction=0.2,
    calibration_seed=42,
    evaluation_subdivisions=2,
)
```

All `HistogramBoost` constructor values not shown remain at committed defaults:
split and leaf L2 regularization both resolve to 1, and total base prior weight
is 1. Each outer fold performs these steps:

1. split only the outer training rows into 80% inner-train and 20% calibration;
2. fit the candidate on inner-train and choose temperature by mean exact CRPS
   on calibration rows;
3. refit a fresh candidate on every outer training row;
4. apply the frozen selected temperature to outer-test probabilities;
5. divide every uniform training bin into two equal-density evaluation bins.

Outer-test targets never select rounds, bins, temperature, support, or any
other hyperparameter. Subdivision preserves the represented density, exact
CRPS, mean, and variance; it only reduces whole-bin quantile-envelope error.
The additional inner fit is included in reported training time.

## Development acceptance

Let `B` be whichever baseline has lower five-fold mean CRPS, selected once for
the dataset rather than per fold. V2 passes only if every condition holds:

1. all 15 expected rows are present, finite, and error-free;
2. `mean_CRPS(V2) / mean_CRPS(B) <= 1.02`;
3. V2 CRPS is no greater than `B` on at least three of five folds;
4. V2 mean absolute 90% coverage error is at most 0.05 and at least 0.02 lower
   than `B`;
5. V2 mean 90% interval score is at most `1.05 * B`;
6. V2 mean RMSE is at most `1.05 * B`.

Passing this development gate allows exactly one frozen confirmation run. It
does not establish an overall win. On `537_houses`, a dataset-level CRPS win
requires ratio below 1.00 and at least four of five fold wins, plus every
guardrail above. Full-suite paired results remain necessary for the product
goal.
