# FormulaBoost

A [varying-coefficient model](https://doi.org/10.1111/j.2517-6161.1993.tb01939.x):
trees learn `θ(Z)`, and a structural input `x` enters only through a
formula you write.

Covariates `Z` determine parameter surfaces `θ(Z)` via trees. A structural
input `x` (spend, dose, time, …) goes into a formula you write:

```
y ≈ f(θ(Z), x)
```

The trees never see `x` as a split feature unless you put it in `Z`. That
is the point: the shape in `x` is constrained by `f`, so the model can
extrapolate in `x` while still letting `θ` vary with `Z`.

## Minimal example

```python
import numpy as np
import openboost as ob

def sales(theta, x):
    a, b = theta
    sigmoid = 1.0 / (1.0 + np.exp(-np.clip(b * x, -30.0, 30.0)))
    return a * x ** sigmoid

model = ob.FormulaBoost(
    formula=sales,
    n_params=2,
    links=("log", "identity"),   # a > 0, b unconstrained
    param_names=("a", "b"),
    precond="full",              # GGN; default
    n_trees=200,
    max_depth=3,
    learning_rate=0.1,
)
model.fit(Z_train, y_train, model_input=x_train)

params = model.predict_params(Z_test)              # dict: a(z), b(z)
yhat = model.predict(Z_test, model_input=x_test)
```

`formula(theta, x)` receives `theta` as a tuple of `K` arrays (already
passed through the links) and `x` as a 1-d array. It must return a 1-d
prediction the same length as `x`.

## Links

Each parameter has a link that maps unconstrained tree scores to the
domain the formula expects.

| Link | Constrained range | Typical use |
|---|---|---|
| `identity` | ℝ | unconstrained coefficients |
| `log` | (0, ∞) | scales, rates |
| `softplus` | (0, ∞) | scales, smoother than `log` |
| `sigmoid` | (0, 1) | fractions, saturations |

`links` must have length `n_params`.

## Preconditioner

FormulaBoost scores the formula with a finite-difference Jacobian `J` and
preconditions with the generalized Gauss-Newton matrix `JᵀJ` (plus
Levenberg–Marquardt damping `damp`).

| `precond` | What it does | When |
|---|---|---|
| `full` (default) | Invert the `K×K` GGN including off-diagonals | Production. Recovers coupled parameters. |
| `diag` | Invert only `diag(JᵀJ)` | Faster; similar predictive RMSE, weaker parameter recovery. |
| `plain` | Raw gradient, no GGN | Debugging only. Diverges on the sales curve. |

XGBoost's custom-objective API accepts only a **diagonal** Hessian. That
is why `full` vs XGBoost-diag is the capability comparison, not a speed
contest.

`damp` (default `1.0`) is added to the GGN diagonal. Increase it if
updates explode; decrease it if training crawls.

## Fit signature

```python
model.fit(
    Z, y, model_input=x,
    sample_weight=None,
    eval_set=[(Z_val, y_val, x_val)],
    callbacks=[ob.EarlyStopping(patience=20)],
    early_stopping_rounds=20,
)
```

`eval_set` entries are always `(X, y, model_input)`, three arrays.

## What to evaluate

On the sales-curve benchmark (200K rows, train `x ∈ [0.25, 2.5]`, extrap
`x ∈ [3, 5]`):

| | test RMSE | extrap RMSE | corr `b(z)` |
|---|--:|--:|--:|
| FormulaBoost `full` | 0.139 | **0.183** | **0.877** |
| FormulaBoost `diag` | 0.130 | 0.181 | 0.730 |
| black-box GBDT | 0.127 | 3.872 | n/a |
| XGBoost custom (diag Hess) | 0.129 | 0.203 | 0.599 |
| global `(a, b)` | 1.641 | 4.396 | n/a |

Honest reading:

- **Extrapolation** is the product claim (~21× vs black-box). The formula
  constrains the shape in `x`; a GBDT that splits on `x` does not.
- **Parameter recovery** of `b(z)` is what `full` GGN buys. If you only
  care about in-sample RMSE, `diag` is enough.
- `plain` is not a baseline you should ship (extrap RMSE ~3.9).

Reproduce: `uv run modal run benchmarks/bench_formula.py`. Full notes:
[Benchmarks](../benchmarks.md).

## Tips

- Keep `K` small. The GGN is `K×K` per row; two to five parameters is the
  intended range.
- Put only **covariates** in `Z`. Put the structural axis in
  `model_input`. Mixing them usually destroys extrapolation.
- Start with a global fit in your head (what would one `(a, b)` be?) so
  you can tell whether the surfaces are recovering anything.
- Shallower trees (`max_depth=3`) and more rounds, same as NaturalBoost.

## Persistence

```python
model.save("formula.joblib")
loaded = ob.FormulaBoost.load("formula.joblib")
```

The formula callable is pickled with the model. Keep it importable under
the same module path if you move files between save and load.
