# Benchmarks

Three gates. All numbers below are from committed Modal runs (A100 for
speed and capability, CPU for UCI quality). This page is the source of
truth that the README summarizes.

## Speed: NaturalBoost vs NGBoost

Heteroscedastic Normal, 80 features, 500 trees, learning rate 0.03,
depth 3. OpenBoost on a Modal A100; NGBoost on CPU (it has no GPU).

| n_train | OpenBoost (A100) | NGBoost (CPU) | speedup | OB NLL | NGB NLL | cov90 |
|--------:|-----------------:|--------------:|--------:|-------:|--------:|------:|
| 45,000 | 3.01s | 1413.93s | 470× | 2.107 | 2.105 | 0.884 |
| 90,000 | 2.21s | 2715.58s | **1229×** | 2.108 | 2.102 | 0.899 |
| 450,000 | 5.40s | n/a | n/a | 2.100 | n/a | 0.901 |
| 900,000 | 6.94s | n/a | n/a | 2.104 | n/a | 0.902 |

NGBoost was not run past 100K (45 minutes at 90K). Quality is tied at
every size both finished. Linear-scale extrapolation of NGBoost to 900K
is ~hours; we do not quote that as a measured speedup.

**Honest reading.** The 1229× is GPU OpenBoost vs CPU NGBoost, because
that is the product comparison. On CPU the two libraries are ~parity
(0.8–1.3× wall-clock, NLL/CRPS/RMSE within ~1%) on the older committed
CPU comparison (`benchmarks/results/ngboost_comparison_20260720.json`).

```bash
# A100 speed + quality
uv run modal run benchmarks/bench_probabilistic.py --suite speed
uv run modal run benchmarks/bench_probabilistic.py::quality

# CPU-only NGBoost comparison (no GPU)
OPENBOOST_BACKEND=cpu uv run --with ngboost python benchmarks/bench_ngboost_comparison.py
```

## Quality: UCI vs NGBoost

NGBoost-paper UCI datasets, 20 paired 80/20 splits, shared 500-tree
budget + patience-50 early stopping on a common val set. NLL, lower is
better. `delta = OB − NGB` (negative = OpenBoost better). `p` is a paired
Wilcoxon.

| dataset | OB NLL | NGB NLL | delta | p | cov90 |
|---|--:|--:|--:|--:|------:|
| boston | 2.679 | 2.639 | +0.040 | 0.57 | 0.894 |
| concrete | 3.128 | 3.135 | −0.007 | 0.60 | 0.848 |
| energy | 1.694 | 1.701 | −0.007 | 0.09 | 0.911 |
| kin8nm | −0.430 | −0.400 | −0.031 | **2e-6** | 0.851 |
| protein | 1.930 | 1.943 | −0.013 | **0.002** | 0.940 |
| wine | 1.028 | 1.031 | −0.003 | 0.13 | 0.879 |
| yacht | 0.814 | 0.828 | −0.014 | 0.73 | 0.877 |
| california | 0.584 | 0.596 | −0.011 | **2e-6** | 0.900 |

Tied-or-better on **8 of the 11** datasets in the suite. Significant wins
on kin8nm, protein, california; no significant loss. boston +0.04 is not
significant. 90% coverage lands in 0.85–0.94.

**Coverage of the suite is incomplete.** `naval_propulsion_plant`,
`power`, and `YearPredictionMSD` are **unmeasured**, not neutral: they
dropped out during an OpenML outage (naval's `data_id` 44898 is a
deactivated version; the name endpoint was returning 503). The rerun that
would fill them in has not been completed, so "tied-or-better" covers 8
datasets and says nothing about the other 3. california is from a local
baseline run, because Modal's figshare egress returned 403.

```bash
uv run modal run benchmarks/bench_probabilistic.py::quality
```

## Capability: FormulaBoost

Sales curve `y = a(z) * x ** sigmoid(b(z) * x)`. Train `x ∈ [0.25, 2.5]`,
extrap `x ∈ [3, 5]` against the noiseless true curve. 300 rounds, depth 3,
lr 0.1. n = 200K:

| | test RMSE | extrap RMSE | corr `b` | fit |
|---|--:|--:|--:|--:|
| FormulaBoost `full` | 0.139 | **0.183** | **0.877** | 17.1s |
| FormulaBoost `diag` | 0.130 | 0.181 | 0.730 | 12.0s |
| FormulaBoost `plain` | 1.410 | 3.931 | 0.652 | 10.3s |
| global `(a, b)` | 1.641 | 4.396 | n/a | n/a |
| black-box GBDT | 0.127 | 3.872 | n/a | 0.7s |
| XGBoost custom (diag Hess) | 0.129 | 0.203 | 0.599 | 19.9s |

Gates (40K and 200K): extrap vs black-box ≥5× (measured **21×**), `full`
beats `plain`, `full` beats global, `full` not worse than XGBoost-diag.

**Honest reading.** The claim that lands is extrapolation (the formula
constrains `x`) and recovery of `b(z)`. Full GGN's edge over diag is
parameter recovery, not test RMSE (`0.181` vs `0.183`). `plain` diverges, so
GGN is load-bearing. XGBoost's custom-objective API is diagonal-only, so
it cannot express the off-diagonal term that buys `b(z)`.

```bash
uv run modal run benchmarks/bench_formula.py
```

## Capability: Weibull AFT

Both `λ(z)` and `k(z)` vary with covariates. ~35% right-censoring. 300
rounds, depth 3. n = 200K:

| | C-index | NLL | cov80 | shape corr | fit |
|---|--:|--:|--:|--:|--:|
| OpenBoost `WeibullAFT` | **0.680** | **0.761** | 0.803 | **0.997** | 5.4s |
| XGBoost `survival:aft` (`extreme`) | 0.672 | 0.830 | 0.852 | n/a (global `k = 1.34`) | 11.7s |
| global constant | 0.500 | 0.896 | 0.810 | n/a | n/a |

C-index is close (ranking follows scale). The NLL gap and the shape
correlation are the capability: XGBoost holds Weibull shape as one
hyperparameter. Coverage of the 80% interval is nearer the nominal 0.80.

```bash
uv run modal run benchmarks/bench_survival.py
```

## What we are not claiming

- A 3900× speedup at 1M rows. That is a linear extrapolation of NGBoost,
  not a measurement.
- That FormulaBoost `full` wins in-sample RMSE. It does not, vs `diag`.
- That OpenBoost is a faster drop-in for XGBoost/LightGBM mean regression.
  It is not; those are optimized C++.
- PGBM as a product competitor. It is a GPU probabilistic reference, not
  a gate.
- Quality parity on the full UCI suite. Three datasets (naval, power,
  YearPredictionMSD) never produced numbers; the claim is 8 of 11.

## JSON artifacts

Reports write to `benchmarks/results/` (gitignored locally; the numbers on
this page are transcribed from the Modal runs recorded in
`tasks/todo.md`). Re-run the commands above to refresh them.
