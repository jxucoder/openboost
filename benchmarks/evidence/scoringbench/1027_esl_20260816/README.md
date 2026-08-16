# ScoringBench `1027_ESL` sentinel — 2026-08-16

This directory freezes OpenBoost's first successful real-dataset shard under
ScoringBench's official five-fold, 3,000-row-cap protocol. It is an integration
and direction-finding result, not a full-suite leaderboard claim.

## Provenance

- GitHub Actions run: [31922702524](https://github.com/jxucoder/openboost/actions/runs/31922702524)
- Actions artifact: `9256929555`
- Actions artifact digest:
  `sha256:0c5176c4a28cd44444f6a086643a01f636b0dcb5e165a4a30d3d992f3e96da97`
- OpenBoost source head: `921c024b23fc4549b251222f35ce64a4b6846505`
- Tested PR merge: `71dd86b7c95245593030ca488981bc2607f02eb5`
- ScoringBench: `a938a667b7839b41e9272929010573410301c0b4`
- Models: OpenBoost NaturalBoost Normal CPU and NGBoost Normal
- Shared parameters: 500 rounds, learning rate 0.01, depth 3, 99 quantiles,
  seed 42

See `openboost_manifest.json` for the complete environment and arguments.
`datasets.json` is ScoringBench's resolved registry. The two Parquet files under
`raw/` contain all fold-level metrics.

## Descriptive result

| Metric (lower is better unless noted) | OpenBoost mean | NGBoost mean | OpenBoost relative | Fold count |
| --- | ---: | ---: | ---: | ---: |
| CRPS | 0.300527 | 0.307714 | -2.34% | 2/5 lower |
| Log score | 0.710078 | 0.675967 | +5.05% | 1/5 lower |
| RMSE | 0.545923 | 0.554927 | -1.62% | 3/5 lower |
| PIT KS statistic | 0.089366 | 0.105493 | -15.29% | 3/5 lower |
| 90% interval score | 2.436058 | 2.737215 | -11.00% | 4/5 lower |
| Fit time (seconds) | 2.111374 | 3.298695 | -35.99% | 5/5 lower |
| 90% coverage (closer to 0.90 is better) | 0.854723 | 0.813718 | — | 5/5 closer |

Mean fit time is 1.56× lower for OpenBoost in this run. The first OpenBoost fold
includes cold Numba compilation, but this artifact does not separately report
cold and warm timing, so it cannot support a general speed claim.

## Interpretation limits

- This is one small real dataset and five correlated CV folds, with no repeated
  seeds or uncertainty interval over the paired differences.
- OpenBoost improves several metrics here but loses log score. CRPS also improves
  on only two individual folds despite its better mean.
- The result says nothing about CUDA or large-data scaling.
- A defensible value claim requires the complete ScoringBench suite, upstream
  review, and a separate multi-size CPU/CUDA extension.
