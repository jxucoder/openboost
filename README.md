# OpenBoost

**GPU gradient boosting for distributional regression.**

Every parameter of `F(y | x)` gets its own tree ensemble, updated with the full
`K×K` natural gradient rather than a diagonal approximation. `FormulaBoost`
extends the same engine to varying-coefficient formulas `y = f(θ(z), x)`.

> 1.0.0rc1. APIs may still move, so install with `--pre` until 1.0.

## Install

```bash
pip install --pre openboost              # core
pip install --pre "openboost[cuda]"      # GPU trees
pip install --pre "openboost[sklearn]"   # sklearn wrappers
```

Python 3.10+. NVIDIA GPU optional (CUDA 11/12).

## Distributional regression

**NaturalBoost** predicts a distribution instead of a point. Every parameter
of `F(y | x)` gets its own tree ensemble, trained by natural gradient.

```python
import openboost as ob

model = ob.NaturalBoostNormal(n_trees=500, max_depth=3, learning_rate=0.03)
model.fit(X_train, y_train)

mean = model.predict(X_test)
lo, hi = model.predict_interval(X_test, alpha=0.1)   # 90% interval
```

**WeibullAFT** takes the same idea to right-censored survival. Both the scale
and the shape vary by covariate, so the hazard shape is per row rather than
one global hyperparameter.

```python
import openboost as ob

model = ob.WeibullAFT(n_trees=300, max_depth=3)
model.fit(Z_train, time_train, event=observed)   # 1 = event, 0 = censored

params = model.predict_params(Z_test)            # {scale, shape}
t_hat = model.predict(Z_test)                    # median time
s = model.predict_survival(Z_test, t=5.0)        # S(5 | z)
```

## Varying-coefficient models

**FormulaBoost** boosts the coefficients of a formula you write. Given
`y = f(θ, x)`, the trees learn `θ(z)` while `x` enters only through `f`.

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

params = model.predict_params(Z_test)              # per-row a(z), b(z)
yhat = model.predict(Z_test, model_input=x_new)
```

## Mean regression

The single-parameter case of the same engine: `GradientBoosting`,
`OpenBoostGAM`, DART, linear-leaf models, and sklearn wrappers. They are here
because they share the trainer and the tree code, not because they beat
XGBoost or LightGBM at plain MSE or logloss. Those are optimized C++ and
should stay your default for point estimates.

## How it compares

Each library optimizes for a different target. NGBoost introduced
natural-gradient distributional boosting and stays close to sklearn on CPU.
XGBoost is the reference for fast mean regression, and its custom-objective
API takes a diagonal Hessian, which is a sound trade for that goal but cannot
represent the off-diagonal coupling between formula parameters; `survival:aft`
likewise holds the Weibull shape fixed across rows. OpenBoost gives up C++
speed on plain regression in exchange for the full metric and an open model
class.

|                             | NGBoost                                   | XGBoost                                | OpenBoost                          |
| --------------------------- | ----------------------------------------- | -------------------------------------- | ---------------------------------- |
| What varies with covariates | Distribution parameters (fixed catalogue) | The mean, or a diagonal custom objective | Distribution or formula parameters |
| Metric                      | Natural gradient                          | Diagonal Hessian                       | Fisher / full GGN                  |
| GPU trees                   | No                                        | Yes                                    | Yes                                |
| Weibull shape `k(z)`        | n/a                                       | Global hyperparameter                  | Per row                            |

## Benchmarks

Early and incomplete, so read them as directional rather than settled.
NaturalBoost matches NGBoost's NLL on the UCI datasets measured so far and
trains in seconds on an A100 at sizes where NGBoost, which is CPU-only, takes
most of an hour. FormulaBoost and WeibullAFT recover parameter surfaces that a
diagonal-Hessian objective cannot. Three UCI datasets have not been measured,
and XGBoostLSS and LightGBMLSS are not in the comparison yet.

Numbers, caveats, and reproduce commands are on the
[benchmarks page](https://jxucoder.github.io/openboost/benchmarks/).

## Documentation

**[jxucoder.github.io/openboost](https://jxucoder.github.io/openboost)**

- [Quickstart](https://jxucoder.github.io/openboost/getting-started/quickstart/)
- [How it works](https://jxucoder.github.io/openboost/user-guide/how-it-works/)
- [NaturalBoost](https://jxucoder.github.io/openboost/user-guide/naturalboost/overview/)
- [FormulaBoost](https://jxucoder.github.io/openboost/user-guide/formulaboost/)
- [Weibull AFT](https://jxucoder.github.io/openboost/user-guide/survival/)
- [Benchmarks](https://jxucoder.github.io/openboost/benchmarks/)
- [API reference](https://jxucoder.github.io/openboost/api/openboost/)



## License

Apache 2.0