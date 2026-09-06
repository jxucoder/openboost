# Distributional and varying-coefficient models

OpenBoost estimates the parameters of a statistical model as functions of
covariates, by gradient boosting.

That is **distributional regression** when the model is a conditional
distribution `F(y | x)`: each parameter (location, scale, shape, ...)
gets its own additive predictor. The statistics literature calls the
GAM version [GAMLSS](https://doi.org/10.1111/j.1467-9876.2005.00510.x);
the boosting version is [gamboostLSS](https://doi.org/10.1214/12-AOAS580)
/ NGBoost. NaturalBoost and WeibullAFT live here.

It is a **varying-coefficient model**
([Hastie & Tibshirani, 1993](https://doi.org/10.1111/j.2517-6161.1993.tb01939.x))
when you supply a structural formula `y ≈ f(θ(z), x)`. The trees learn
`θ(z)`; `x` enters only through `f`. That is FormulaBoost.

```
z  --trees-->  F (n × K raw scores)  --link-->  θ  --f-->  prediction  --loss-->  scalar
```

`z` are the covariates the trees split on. `θ = (θ₁, …, θₖ)` is the
parameter vector of the statistical model. Each `θᵢ` is its own ensemble.

The per-round update is a (damped) generalized Gauss-Newton / natural
gradient step:

```
dᵢ = (Jᵢᵀ Mᵢ Jᵢ + λI)⁻¹ gᵢ     # K × K per sample, K small
```

Models are configurations of `f`, the loss, and the metric `M`.

| Model | Statistical object | Loss | Metric `M` | `K` |
|---|---|---|---|---|
| [NaturalBoost](naturalboost/overview.md) | Conditional distribution | NLL | analytic Fisher | # distribution params |
| [WeibullAFT](survival.md) | Censored Weibull AFT | censored NLL | expected Fisher | 2 (scale, shape) |
| [FormulaBoost](formulaboost.md) | Varying-coefficient formula | MSE | GGN (`plain` / `diag` / `full`) | # formula params |
| `GradientBoosting` | Conditional mean | MSE / logloss / … | scalar Hessian | 1 |

`K = 1` is ordinary mean regression. `K > 1` is the point of this library.

## What this is not

Not a faster drop-in for XGBoost/LightGBM mean regression. Those are
optimized C++ and should stay the default for MSE/logloss.

Not only "faster NGBoost." NGBoost is distributional regression for a
fixed set of families, with exact-split trees, on CPU. OpenBoost does
the same job with histogram trees and a GPU path, and also fits
varying-coefficient formulas and a censored Weibull whose shape depends
on covariates, neither of which NGBoost can write down.

Quality and performance comparisons require workload-specific evidence:
[Benchmarks](../benchmarks.md).

## Why the metric matters

Raw gradients of a multi-parameter likelihood can have different scales.
FormulaBoost offers plain, diagonal and full GGN preconditioning; compare
convergence and parameter recovery on the intended formula. Full GGN retains
cross-parameter terms but does not guarantee better held-out error.

WeibullAFT uses an expected-Fisher step rather than the observed Hessian.
These implementation choices require task-level evaluation; see the
[benchmark protocol](../benchmarks.md).

## Shared training API

NaturalBoost, FormulaBoost, and WeibullAFT share one trainer:

```python
model.fit(
    X, y,                          # FormulaBoost also needs model_input=x
    eval_set=[...],                # early stopping / logging
    callbacks=[ob.EarlyStopping(patience=20)],
    early_stopping_rounds=20,
)
params = model.predict_params(X)   # dict of per-row parameter arrays
```

- NaturalBoost / WeibullAFT: `eval_set` is `(X_val, y_val)` (plus `event`
  for AFT).
- FormulaBoost: `eval_set` is `(X_val, y_val, model_input_val)`.

GPU: histogram trees run on CUDA when `openboost[cuda]` is installed.
NaturalBoost Normal/Poisson have device gradient kernels. FormulaBoost GGN
is currently host-side.

## Choosing a model

| You have | Use |
|---|---|
| `y` should be a distribution given `X` | `NaturalBoost*` |
| A known curve `y = f(θ, x)` with `θ` depending on other features | `FormulaBoost` |
| Right-censored times, Weibull, shape should depend on covariates | `WeibullAFT` |
| A custom NLL that is still a distribution | `NaturalBoost` + [custom distribution](naturalboost/custom-distributions.md) |
| Ordinary mean regression | `GradientBoosting` (or XGBoost / LightGBM) |

## Next

- [NaturalBoost](naturalboost/overview.md)
- [FormulaBoost](formulaboost.md)
- [Weibull AFT](survival.md)
- [Benchmarks](../benchmarks.md)
