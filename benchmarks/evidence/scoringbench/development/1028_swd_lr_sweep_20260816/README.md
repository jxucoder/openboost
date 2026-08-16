# 1028_SWD learning-rate development sweep

This is tuning evidence, not held-out or leaderboard evidence. The clean
five-model baseline used 500 OpenBoost/NGBoost rounds at learning rate 0.01.
After inspecting that result, one OpenBoost-only candidate changed only the
learning rate to 0.03. Both runs used the same ScoringBench commit, dataset,
seed, five folds, sample cap, Normal distribution, depth, regularization, and
99-quantile representation.

The baseline disproved a broad win: OpenBoost had the best mean 90% interval
score, absolute coverage error, and PIT KS, but its mean CRPS was 4.9% worse
than native XGBoost quantile, 1.7% worse than XGBoostLSS, and 2.0% worse than
NGBoost. Its CRPS lost to native XGBoost quantile on all five folds.

| OpenBoost setting | CRPS | RMSE | 90% coverage | Coverage error | 90% interval score | PIT KS | Sharpness |
|---|---:|---:|---:|---:|---:|---:|---:|
| lr=0.01, 500 rounds | **0.3551** | **0.6260** | **0.8910** | **0.0310** | **2.5598** | **0.0873** | 0.5801 |
| lr=0.03, 500 rounds | 0.3566 | 0.6278 | 0.8680 | 0.0320 | 2.6811 | 0.0959 | **0.5490** |

Increasing the learning rate sharpened the distribution by 5.4%, but mean
CRPS worsened by 0.44%, interval score by 4.74%, PIT KS by 9.82%, and coverage
moved farther below 90%. The candidate improved CRPS on only two of five
paired folds and interval score on one. This falsifies the hypothesis that the
current CRPS gap is primarily caused by an update budget that is too small; the
0.03 default used by an older NGBoost comparison should not be copied into the
ScoringBench wrapper.

The apparent 11.6% fit-time change is from separate Actions processes and is
not a speed result. ScoringBench reconstructed log score and CRLS are retained
in the raw Parquet files but excluded from the decision because of the known
finite-support and model-specific-grid comparability problems.

Baseline: [run 31926255124](https://github.com/jxucoder/openboost/actions/runs/31926255124), artifact `9258026048`, digest
`sha256:495ac3c8c02c0846423131c4636dfde05f0da872b637bca2be9493b3c02aa8b4`.

Candidate: [run 31926664340](https://github.com/jxucoder/openboost/actions/runs/31926664340), artifact `9258115955`, digest
`sha256:b3b77cbf691bef574b0a27c897b8a92a22e90a4c158f0f7e3ae556400d60e599`.

`summary.json` records exact unrounded effects and SHA-256 hashes for every
copied raw/result/provenance file. The next diagnostic must separate point-mean
error from scale calibration, then test post-fit scale calibration or a
different scale objective without changing the consumed `1027_ESL` shard.
