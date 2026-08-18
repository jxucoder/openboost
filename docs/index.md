# OpenBoost

<p align="center" style="font-size: 1.4em;">
  <strong>GPU gradient boosting for distributional regression.</strong>
</p>

<p align="center">
  Every parameter of <code>F(y | x)</code> gets its own tree ensemble, updated
  with the full <code>K&times;K</code> natural gradient rather than a diagonal
  approximation. <code>FormulaBoost</code> extends the same engine to
  varying-coefficient formulas <code>y = f(θ(z), x)</code>.
</p>

<p align="center">
  <a href="getting-started/quickstart/">Quickstart</a> •
  <a href="user-guide/how-it-works/">How it works</a> •
  <a href="user-guide/formulaboost/">FormulaBoost</a> •
  <a href="user-guide/survival/">Weibull AFT</a> •
  <a href="benchmarks/">Benchmarks</a> •
  <a href="api/openboost/">API</a>
</p>

---

## Why this exists

[Distributional regression](https://doi.org/10.1111/j.1467-9876.2005.00510.x)
(GAMLSS) fits every parameter of `F(y | x)` as a function of covariates,
not just the mean. NGBoost does that with boosting, on CPU, for a few
built-in families.

A [varying-coefficient model](https://doi.org/10.1111/j.2517-6161.1993.tb01939.x)
does the same for a structural formula `y ≈ f(θ(z), x)`. XGBoost custom
objectives accept only a **diagonal** Hessian, so they cannot represent
the off-diagonal coupling in `θ`. XGBoost `survival:aft` holds the
Weibull shape as one global hyperparameter.

OpenBoost fits both classes, censoring included, with histogram trees, a
full per-sample Fisher or GGN metric, and a GPU path.

| | NGBoost | XGBoost | OpenBoost |
|---|---|---|---|
| What varies with covariates | Distribution parameters (fixed catalogue) | The mean, or a diagonal custom objective | Distribution or formula parameters |
| Metric | Natural gradient | Diagonal Hessian only | Fisher / full GGN |
| GPU trees | No | Yes (mean / AFT location) | Yes |
| Covariate-dependent Weibull shape `k(z)` | n/a | No (global hyperparameter) | Yes |
| Formula off-diagonals | n/a | Inexpressible | Full GGN |

For ordinary mean regression, use XGBoost or LightGBM. They are faster
C++.

## The models

=== "NaturalBoost"

    Distributional regression. Predict `F(y | x)`, not a point.

    ```python
    import openboost as ob

    model = ob.NaturalBoostNormal(n_trees=500, max_depth=3, learning_rate=0.03)
    model.fit(X_train, y_train)
    mean = model.predict(X_test)
    lo, hi = model.predict_interval(X_test, alpha=0.1)   # 90% interval
    ```

=== "FormulaBoost"

    Varying-coefficient model. Write `y = f(θ, x)`; trees learn `θ(z)`.

    ```python
    import numpy as np
    import openboost as ob

    def sales(theta, x):
        a, b = theta
        return a * x ** (1.0 / (1.0 + np.exp(-b * x)))

    model = ob.FormulaBoost(
        formula=sales, n_params=2, links=("log", "identity"),
        param_names=("a", "b"), precond="full",
    )
    model.fit(Z_train, y_train, model_input=x_train)
    params = model.predict_params(Z_test)          # per-row a(z), b(z)
    yhat = model.predict(Z_test, model_input=x_new)
    ```

=== "WeibullAFT"

    Censored Weibull AFT; both scale and shape vary with `z`.

    ```python
    import openboost as ob

    model = ob.WeibullAFT(n_trees=300, max_depth=3)
    model.fit(Z_train, time_train, event=observed)  # 1 = event, 0 = censored
    params = model.predict_params(Z_test)           # {scale, shape}
    t_hat = model.predict(Z_test)                   # median time
    s = model.predict_survival(Z_test, t=5.0)       # S(5 | z)
    ```

Same trainer, same tree engine. Mean regression is the single-parameter case
of it: `GradientBoosting`, `OpenBoostGAM`, DART, linear-leaf models, and
sklearn wrappers ship in the same package.

## Benchmarks

Early and incomplete, so read them as directional rather than settled.

| Gate | Where it stands |
|---|---|
| Speed vs NGBoost | Seconds on an A100 at sizes where NGBoost, which is CPU-only, takes most of an hour. On CPU the two are near parity, so this is a claim about the GPU tree path. |
| Quality vs NGBoost | Tied or better NLL on the 8 UCI datasets measured, over 20 paired splits. Three datasets are still unmeasured. |
| Capability | FormulaBoost and WeibullAFT recover parameter surfaces that a diagonal-Hessian objective cannot express. |

XGBoostLSS and LightGBMLSS are the nearest alternatives and are not yet in the
comparison. Numbers, caveats, and reproduce commands:
[Benchmarks](benchmarks.md).

## Who this is for

- Insurance pricing, energy/demand, credit risk: you need `F(y | x)`, not a point
- Curve / dose / saturation models where the formula is the product and `θ(z)` is what you ship
- Survival analysis where the Weibull shape should move with covariates
- Anyone who has been hand-rolling an XGBoost custom objective and hitting the diagonal-Hessian wall

Not the right tool if you want the fastest possible MSE/logloss GBDT. Use
XGBoost or LightGBM for that.

## Install

```bash
pip install --pre openboost
pip install --pre "openboost[cuda]"
```

Without `--pre`, pip installs the older stable release rather than the
current 1.0 release candidate. See [Installation](getting-started/installation.md).

## License

Apache 2.0
