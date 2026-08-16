# CRPS distribution experiment v1

This protocol was frozen before loading either dataset or observing any result
on them. It separates architecture development from confirmation after the
`1027_ESL` and `1028_SWD` diagnostics were consumed.

## Frozen data roles

- Development: `197_cpu_act`, a continuous regression dataset with 8,192 rows
  and 21 machine-performance features.
- Untouched confirmation: `537_houses`, a continuous housing/population
  regression dataset with 8 numeric features. Do not load, validate, or run
  this entry until one candidate implementation and configuration have been
  frozen after the development result. The launcher enforces this registry
  role unless the confirmation run explicitly supplies `--allow-confirmation`.

Both URLs in `crps_distribution_v1.json` point to PMLB commit
`9cc9017958f2d8284e62d8bc54b77cb6fa1e9592`. The registry also records the
expected SHA-256 of each compressed source file. A run must fail closed if the
downloaded bytes do not match. The mutable PMLB `master` URLs are ineligible
for this experiment.

## Protocol

- ScoringBench commit:
  `a938a667b7839b41e9272929010573410301c0b4`.
- Five folds, one repeat, seed 42, sample cap 3,000, CPU execution.
- Compare the frozen OpenBoost candidate with ScoringBench's native XGBoost
  multi-quantile model and CatBoost MultiQuantile model.
- XGBoost: 100 rounds and 50 quantiles, seed 42, two threads.
- CatBoost: 1,000 rounds and 99 quantiles, seed 42, two threads.
- Use an empty output directory. Exactly 15 finite result rows must exist.
- CRPS is primary. Log score and CRLS are excluded because the pinned evaluator
  does not put different finite-support predictions on a common support.

The candidate family is a non-parametric histogram distribution trained by
direct CRPS gradients with one shared tree and vector leaves per round. Only a
positive-semidefinite Gauss--Newton curvature is eligible; the earlier
absolute-value transform of an indefinite exact diagonal is rejected. The
candidate's exact public API and hyperparameters must be committed before the
development dataset is loaded.

The frozen candidate is `openboost_histogram_cpu`, backed by:

```python
openboost.HistogramBoost(
    n_distribution_bins=50,
    n_trees=100,
    learning_rate=0.05,
    max_depth=6,
    n_feature_bins=254,
    curvature_scale=1.0,
)
```

All other constructor values remain at the committed defaults. The wrapper
passes its regular shared grid and PMF directly to ScoringBench as a natively
gridded prediction; it does not derive or regrid quantiles.

## Development acceptance

Let `B` be whichever of XGBoost quantile and CatBoost MultiQuantile has the
lower five-fold mean CRPS. Choose `B` once per dataset, never separately for
each fold. A candidate passes only if every condition holds:

1. all 15 expected rows are present, finite, and error-free;
2. `mean_CRPS(candidate) / mean_CRPS(B) <= 1.02`;
3. candidate CRPS is no greater than `B` on at least three of five folds;
4. candidate mean absolute 90% coverage error is at most 0.05 and at least
   0.02 lower than `B`;
5. candidate mean 90% interval score is at most `1.05 * B`;
6. candidate mean RMSE is at most `1.05 * B`.

Passing development allows one frozen run on `537_houses`; it is not evidence
of a general win.

## Confirmation language

On `537_houses`, OpenBoost may be described as having lower CRPS on that
dataset only when its mean CRPS ratio to `B` is below 1.00 and it wins at least
four of five folds. A ratio at or below 1.02 that also satisfies the guardrails
is only calibrated non-inferiority. Neither outcome is an overall or SOTA
claim. The full ScoringBench suite remains the product acceptance test.
