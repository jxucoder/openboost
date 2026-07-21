# Linear Leaf GBDT

Trees with linear models in the leaves instead of constant values. Better for extrapolation and smooth relationships.

## Why Linear Leaves?

Standard trees predict constant values in each leaf. Linear Leaf GBDT fits a linear model in each leaf, which:

- Captures linear trends within regions
- Extrapolates better outside training data
- Needs fewer trees for smooth relationships

## Basic Usage

```python
import openboost as ob

model = ob.LinearLeafGBDT(
    n_trees=100,
    max_depth=4,
    learning_rate=0.1,
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Parameters

`LinearLeafGBDT` has its own parameter set (it is not a `GradientBoosting`
subclass, so standard parameters like `subsample` or `colsample_bytree` are
not available):

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_trees` | int | 100 | Number of boosting rounds |
| `max_depth` | int | 4 | Maximum tree depth (typically 3-4, shallower than standard) |
| `learning_rate` | float | 0.1 | Shrinkage factor |
| `loss` | str/callable | `'mse'` | Loss function (`'mse'`, `'mae'`, `'huber'`, or callable) |
| `min_samples_leaf` | int | 20 | Minimum samples to fit a linear model in a leaf |
| `reg_lambda_tree` | float | 1.0 | L2 regularization for tree splits |
| `reg_lambda_linear` | float | 0.1 | Ridge regularization for leaf linear models |
| `max_features_linear` | int/str/None | `'sqrt'` | Features per leaf model: `None` (all), `'sqrt'`, `'log2'`, or an int |
| `n_bins` | int | 256 | Number of bins for histogram building |

## When to Use

| Situation | Recommendation |
|-----------|----------------|
| Data has linear trends | LinearLeafGBDT |
| Need extrapolation | LinearLeafGBDT |
| Purely nonlinear data | Standard GBDT |
| Maximum speed | Standard GBDT |

## Example

```python
import numpy as np
import openboost as ob

# Data with linear trend + nonlinear pattern
X = np.random.randn(1000, 5).astype(np.float32)
y = 2 * X[:, 0] + np.sin(X[:, 1] * 3) + 0.1 * np.random.randn(1000)
y = y.astype(np.float32)

# Compare models
gbdt = ob.GradientBoosting(n_trees=100, max_depth=6)
linear_leaf = ob.LinearLeafGBDT(n_trees=100, max_depth=4)

gbdt.fit(X[:800], y[:800])
linear_leaf.fit(X[:800], y[:800])

# Evaluate
gbdt_rmse = np.sqrt(np.mean((gbdt.predict(X[800:]) - y[800:])**2))
ll_rmse = np.sqrt(np.mean((linear_leaf.predict(X[800:]) - y[800:])**2))

print(f"GBDT RMSE: {gbdt_rmse:.4f}")
print(f"LinearLeaf RMSE: {ll_rmse:.4f}")
```

## Validation, Callbacks, and Early Stopping

`LinearLeafGBDT.fit` accepts the same training-control arguments as
`GradientBoosting`:

```python
model = ob.LinearLeafGBDT(n_trees=500, max_depth=4)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],          # a single bare (X, y) tuple also works
    callbacks=[ob.Logger(period=25)],
    early_stopping_rounds=20,   # sugar for EarlyStopping(patience=20, restore_best=True)
)

model.evals_result_     # {'eval_0': {'mse': [...]}} — per-round history per eval set
model.best_iteration_   # set when early stopping is used
model.best_score_
```

Details:

- The eval metric is **MSE on raw predictions**, recorded per round for every
  eval set under keys `'eval_0'`, `'eval_1'`, ... (`evals_result_` is `{}`
  when no `eval_set` is passed).
- Callbacks (`EarlyStopping`, `Logger`, `HistoryCallback`, ...) receive the
  **last** eval set's MSE as `val_loss`.
- `early_stopping_rounds=N` stops after `N` rounds without improvement and
  restores the model to (and sets `best_iteration_` / `best_score_` at) the
  best round.

## sklearn Wrapper

```python
from openboost import OpenBoostLinearLeafRegressor

model = OpenBoostLinearLeafRegressor(n_estimators=100, max_depth=4)
model.fit(X_train, y_train)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```
