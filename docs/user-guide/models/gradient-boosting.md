# Gradient Boosting

The core gradient boosting model for regression and binary classification.

The examples on this page share this setup:

```python
import numpy as np
import openboost as ob

rng = np.random.default_rng(0)
X = rng.standard_normal((1000, 8)).astype(np.float32)
y = (X[:, 0] - 2.0 * X[:, 1] + 0.1 * rng.standard_normal(1000)).astype(np.float32)
X_train, y_train = X[:700], y[:700]
X_val, y_val = X[700:850], y[700:850]
X_test, y_test = X[850:], y[850:]
```

## Basic Usage

```python
model = ob.GradientBoosting(
    n_trees=100,
    max_depth=6,
    learning_rate=0.1,
    loss='mse',
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_trees` | int | 100 | Number of boosting iterations |
| `max_depth` | int | 6 | Maximum depth of each tree |
| `learning_rate` | float | 0.1 | Step size shrinkage |
| `loss` | str/callable | 'mse' | Loss function |
| `min_child_weight` | float | 1.0 | Minimum sum of hessian in a leaf |
| `reg_lambda` | float | 1.0 | L2 regularization |
| `subsample` | float | 1.0 | Row subsampling ratio |
| `colsample_bytree` | float | 1.0 | Column subsampling ratio |
| `n_bins` | int | 256 | Number of histogram bins |
| `growth` | str | `'levelwise'` | Tree growth strategy: `'levelwise'`, `'leafwise'`, or `'symmetric'` |
| `max_leaves` | int/None | None | Max leaves per tree for `'leafwise'` growth (defaults to `2**max_depth`) |
| `random_state` | int/None | None | Seed for reproducible training |

## Loss Functions

| Loss | Use Case |
|------|----------|
| `'mse'` | Regression (default) |
| `'mae'` | Robust regression |
| `'huber'` | Outlier-robust regression |
| `'logloss'` | Binary classification |
| `'quantile'` | Quantile regression |
| Custom callable | Your own loss |

### Custom Loss Function

```python
def quantile_loss(pred, y, tau=0.9):
    residual = y - pred
    grad = np.where(residual > 0, -tau, 1 - tau)
    hess = np.ones_like(pred)
    return grad, hess

model = ob.GradientBoosting(n_trees=100, loss=quantile_loss)
model.fit(X_train, y_train)
```

## Training with Validation

```python
model = ob.GradientBoosting(n_trees=500, max_depth=6)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[
        ob.EarlyStopping(patience=10),
        ob.Logger(period=10),
    ],
)
```

## Feature Importance

```python
model.fit(X_train, y_train)

# Compute importance (pass the fitted model, not model.trees_)
importance = ob.compute_feature_importances(model)
print(importance)
```

<!-- docs-ci: skip -->
```python
# Plot (requires matplotlib)
feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
ob.plot_feature_importances(model, feature_names)
```

## Growth Strategies

```python
# Level-wise (XGBoost-style, default)
model = ob.GradientBoosting(growth='levelwise')

# Leaf-wise (LightGBM-style); cap leaf count with max_leaves
model = ob.GradientBoosting(growth='leafwise', max_leaves=31)

# Symmetric/Oblivious (CatBoost-style)
model = ob.GradientBoosting(growth='symmetric')
```

## API Reference

::: openboost.GradientBoosting
    options:
      show_root_heading: true
      members:
        - fit
        - predict
        - save
        - load
