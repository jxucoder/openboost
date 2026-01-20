# Gradient Boosting

The core gradient boosting model for regression and binary classification.

## Basic Usage

```python
import openboost as ob

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
        ob.Logger(every=10),
    ],
)
```

## Feature Importance

```python
model.fit(X_train, y_train)

# Compute importance
importance = ob.compute_feature_importances(model.trees_)

# Plot
ob.plot_feature_importances(model.trees_, feature_names)
```

## Growth Strategies

```python
# Level-wise (XGBoost-style, default)
model = ob.GradientBoosting(growth='levelwise')

# Leaf-wise (LightGBM-style)
model = ob.GradientBoosting(growth='leafwise')

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
