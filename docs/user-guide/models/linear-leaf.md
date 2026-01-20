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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `ridge_lambda` | float | 1.0 | Ridge regularization for leaf linear models |

Plus all standard `GradientBoosting` parameters.

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

## sklearn Wrapper

```python
from openboost import OpenBoostLinearLeafRegressor

model = OpenBoostLinearLeafRegressor(n_estimators=100, max_depth=4)
model.fit(X_train, y_train)
print(f"R² Score: {model.score(X_test, y_test):.4f}")
```
