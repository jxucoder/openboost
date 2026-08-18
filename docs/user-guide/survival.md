# Weibull AFT

Boosted Weibull accelerated-failure-time model with right censoring.

Both the scale `λ(z)` and the shape `k(z)` are boosting ensembles over
covariates `z`, trained on a right-censored negative log-likelihood.
That is the capability NGBoost and XGBoost AFT do not have:

- NGBoost has no censored likelihood.
- XGBoost `survival:aft` learns a location and holds the distribution
  scale as **one global hyperparameter**, so every row shares the same
  Weibull shape.

## Minimal example

```python
import openboost as ob

model = ob.WeibullAFT(n_trees=300, max_depth=3, learning_rate=0.1)
model.fit(Z_train, time_train, event=observed)   # event: 1 seen, 0 censored

params = model.predict_params(Z_test)            # {scale, shape} per row
t_hat = model.predict(Z_test)                    # median survival time
q90 = model.predict_quantile(Z_test, q=0.9)
s = model.predict_survival(Z_test, t=12.0)       # S(12 | z)
nll = model.nll(Z_test, time_test, event=observed_test)
```

`y` is observed time and must be strictly positive. `event` is 1 for an
observed failure and 0 for right-censored. If `event` is omitted, every
row is treated as observed.

## Likelihood and metric

Weibull survival function:

```
S(t | z) = exp( − (t / λ(z))^{k(z)} )
```

Trees produce unconstrained scores `(u, v)`; links map them to
`λ = exp(u)`, `k = exp(v)`. The trainer steps in that unconstrained space
using the **expected Fisher information** of the censored Weibull, not
the observed Hessian. The observed scale term `k² z` blows up when `λ` is
wrong and freezes the update. The expected Fisher does not.

`damp` (default `1.0`) is Levenberg–Marquardt damping on that `2×2`
matrix. Increase it if NLL spikes in the first rounds.

## Fit signature

```python
model.fit(
    Z, time, event=observed,
    eval_set=[(Z_val, time_val, event_val)],
    callbacks=[ob.EarlyStopping(patience=20)],
    early_stopping_rounds=20,
)
```

`eval_set` entries are `(X, y)` or `(X, y, event)`. The logged metric is
censored NLL.

## Predictions

| Method | Returns |
|---|---|
| `predict_params(X)` | `dict` with `scale` (`λ`) and `shape` (`k`), shape `(n,)` |
| `predict(X)` | median time, `λ (ln 2)^{1/k}` |
| `predict_median(X)` | same as `predict` |
| `predict_quantile(X, q)` | time at which `P(T ≤ t) = q` |
| `predict_survival(X, t)` | `S(t \| z)` for a scalar or per-row `t` |
| `nll(X, y, event=)` | mean censored negative log-likelihood |

## vs XGBoost `survival:aft`

Synthetic DGP where **both** `λ(z)` and `k(z)` vary, ~35% right-censoring,
200K rows, 300 rounds:

| | C-index | NLL | 80% coverage | shape corr | fit |
|---|--:|--:|--:|--:|--:|
| OpenBoost `WeibullAFT` | **0.680** | **0.761** | 0.803 | **0.997** | 5.4s |
| XGBoost `survival:aft` (`extreme`) | 0.672 | 0.830 | 0.852 | n/a (global `k=1.34`) | 11.7s |
| global constant | 0.500 | 0.896 | 0.810 | n/a | n/a |

C-index is close, since ranking mostly follows the scale. The NLL gap and the
shape correlation are the capability: OpenBoost recovers `k(z)`, XGBoost
cannot represent it. Coverage of the 80% interval is nearer the nominal
0.80 (XGBoost over-covers).

Reproduce: `uv run modal run benchmarks/bench_survival.py`. Notes:
[Benchmarks](../benchmarks.md).

## Tips

- Times must be `> 0`. Shift or clip before fitting.
- `event` dtype does not matter as long as it is 0/1.
- Shallower trees (`max_depth=3`) and more rounds, same as NaturalBoost.
- If NLL diverges, raise `damp` or lower `learning_rate` before adding
  trees.

## Persistence

```python
model.save("aft.joblib")
loaded = ob.WeibullAFT.load("aft.joblib")
```
