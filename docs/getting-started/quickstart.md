# Quickstart

Distributional regression and a varying-coefficient formula in a few
minutes. Each example is self-contained.

## NaturalBoost: a distribution, not a point

```python
import numpy as np
import openboost as ob

rng = np.random.default_rng(0)
X = rng.standard_normal((2000, 8)).astype(np.float32)
noise = 0.4 + np.abs(X[:, 0])
y = (2.0 * X[:, 0] + X[:, 1] + noise * rng.standard_normal(2000)).astype(np.float32)
X_train, X_test = X[:1600], X[1600:]
y_train, y_test = y[:1600], y[1600:]

model = ob.NaturalBoostNormal(n_trees=200, max_depth=3, learning_rate=0.05)
model.fit(X_train, y_train)

mean = model.predict(X_test)
lo, hi = model.predict_interval(X_test, alpha=0.1)
print(f"90% coverage: {np.mean((y_test >= lo) & (y_test <= hi)):.1%}")
```

Pick a family that matches the data (`NaturalBoostLogNormal`, `Gamma`,
`Poisson`, `Tweedie`, `NegBin`, `StudentT`). See
[distributions](../user-guide/naturalboost/distributions.md).

## FormulaBoost: boost the parameters of a formula

Features `Z` learn parameter surfaces; a structural input `x` (spend, dose,
time) goes into a formula you write.

```python
import numpy as np
import openboost as ob

def sales(theta, x):
    a, b = theta
    return a * x ** (1.0 / (1.0 + np.exp(-np.clip(b * x, -30.0, 30.0))))

rng = np.random.default_rng(0)
n = 4000
Z = rng.uniform(0, 1, (n, 4)).astype(np.float32)
x = rng.uniform(0.3, 2.5, n)
a = np.exp(0.4 + 0.8 * Z[:, 0] - 0.5 * Z[:, 1])
b = 0.6 + 1.5 * Z[:, 2]
y = sales((a, b), x) + 0.05 * rng.standard_normal(n)

model = ob.FormulaBoost(
    formula=sales, n_params=2, links=("log", "identity"),
    param_names=("a", "b"), precond="full",
    n_trees=80, max_depth=3, learning_rate=0.1,
)
model.fit(Z[:3200], y[:3200], model_input=x[:3200])
params = model.predict_params(Z[3200:])
print(params["a"][:5], params["b"][:5])
```

`precond="full"` (default) is the GGN preconditioner. `plain` (raw
gradient) diverges on this formula. Details:
[FormulaBoost](../user-guide/formulaboost.md).

## WeibullAFT: survival with a per-row shape

```python
import numpy as np
import openboost as ob

rng = np.random.default_rng(0)
n = 3000
Z = rng.standard_normal((n, 6)).astype(np.float32)
lam = np.exp(0.5 + 0.8 * Z[:, 0])
k = np.exp(0.2 + 0.6 * Z[:, 1])
u = rng.random(n)
time = lam * ((-np.log(u)) ** (1.0 / k))
event = (rng.random(n) > 0.3).astype(np.float64)  # ~30% right-censored

model = ob.WeibullAFT(n_trees=150, max_depth=3, learning_rate=0.1)
model.fit(Z[:2400], time[:2400], event=event[:2400])
params = model.predict_params(Z[2400:])
t_hat = model.predict(Z[2400:])            # median time
s = model.predict_survival(Z[2400:], t=1.0)
print(params["scale"][:3], params["shape"][:3], t_hat[:3])
```

XGBoost's `survival:aft` cannot vary Weibull shape with covariates.
[Weibull AFT](../user-guide/survival.md).

## Point-estimate GBDT (also here)

```python
model = ob.GradientBoosting(n_trees=100, max_depth=6, loss="mse")
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

Binary classification uses `loss="logloss"`. Multi-class uses
`ob.MultiClassGradientBoosting`. sklearn wrappers
(`OpenBoostRegressor`, `OpenBoostClassifier`,
`OpenBoostDistributionalRegressor`) work with `GridSearchCV` and
`Pipeline`. See [Gradient Boosting](../user-guide/models/gradient-boosting.md)
and [sklearn integration](../user-guide/sklearn-integration.md).

## Callbacks, GPU, persistence

```python
model = ob.NaturalBoostNormal(n_trees=500, max_depth=3)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[ob.EarlyStopping(patience=20), ob.Logger(period=20)],
)
model.save("model.joblib")
loaded = ob.NaturalBoostNormal.load("model.joblib")
```

GPU trees are automatic when CUDA is installed
(`pip install --pre "openboost[cuda]"`). Force a backend with
`ob.set_backend("cuda")` or `OPENBOOST_BACKEND=cuda`. See
[GPU setup](gpu-setup.md).

## Next

- [How it works](../user-guide/how-it-works.md): distributional regression and varying-coefficient models
- [Uncertainty tutorial](../tutorials/uncertainty.md): intervals, sampling, NLL
- [Benchmarks](../benchmarks.md): speed, quality, capability numbers
- [Custom distributions](../user-guide/naturalboost/custom-distributions.md)
