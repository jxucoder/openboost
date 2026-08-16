# 1028_SWD corrected numeric-binning diagnostic

This is development/tuning evidence, not held-out or leaderboard evidence. It
reruns the frozen OpenBoost NLL configuration after commit `236c2df` corrected
numeric binning so that `m` cut edges retain all `m + 1` intervals. The prior
implementation silently merged the highest interval into its predecessor.

The candidate used 500 rounds, learning rate 0.01, depth 3, seed 42, five
folds, sample cap 3000, 99 quantiles, and pinned ScoringBench commit
`a938a667`. Strong-baseline rows come from the already-frozen, same-dataset,
same-split artifact in `../1028_swd_lr_sweep_20260816/baseline_lr_001/`.

| Model | CRPS | RMSE | 90% coverage error | 90% interval score | PIT KS | Sharpness |
|---|---:|---:|---:|---:|---:|---:|
| OpenBoost before fix | 0.355075 | 0.626043 | 0.0310 | 2.559809 | **0.087349** | 0.580131 |
| **OpenBoost after fix** | **0.347845** | **0.615779** | **0.0290** | **2.509361** | 0.098027 | 0.560487 |
| XGBoostLSS Gaussian | 0.349289 | 0.617750 | 0.0410 | 2.670105 | 0.100551 | 0.526127 |
| NGBoost Gaussian | 0.348185 | 0.616315 | 0.0320 | 2.572281 | 0.099395 | 0.542749 |
| XGBoost quantile | 0.338359 | 0.636753 | 0.2670 | 2.788130 | 0.302062 | 0.505502 |
| CatBoost quantile | 0.336377 | 0.635947 | 0.2380 | 2.825541 | 0.172461 | **0.491907** |

Relative to the pre-fix OpenBoost artifact, corrected binning improves CRPS by
2.04% (four of five folds), RMSE by 1.64%, interval score by 1.97%, coverage
error by 6.45%, and sharpness by 3.39%. PIT KS worsens by 12.22%, so the change
is not uniformly better on every diagnostic, although it fixes an unambiguous
representation bug.

On this consumed development dataset, corrected OpenBoost has 0.41% lower mean
CRPS than XGBoostLSS and wins four of five paired folds. It also improves RMSE,
coverage error, interval score, and PIT KS. Against NGBoost, mean CRPS is only
0.10% lower and OpenBoost wins two of five folds; treat that as parity, not a
win.

The stated goal is still unmet. Corrected OpenBoost CRPS remains 2.80% worse
than native XGBoost quantile and 3.41% worse than CatBoost quantile, winning
only one of five folds against each. OpenBoost is much better calibrated on
this dataset and has lower interval score and RMSE, but those guardrails do not
erase the pre-registered CRPS loss. The next candidate must target the
remaining distribution-shape/quantile gap and then be evaluated on new data.

Run: [31928677396](https://github.com/jxucoder/openboost/actions/runs/31928677396),
artifact `9258714239`, digest
`sha256:c75eea2d93bf10d5de1ae48a8ab37eed66c1dc1c326a9810ab8dc808c4656836`.
The run completed 5/5 rows from clean source `236c2df`; the manifest records a
clean pinned ScoringBench checkout and the full Linux environment. Fit times
from separate Actions processes are not used as speed evidence.
