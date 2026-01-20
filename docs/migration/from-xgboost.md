# Migrating from XGBoost to OpenBoost

This guide helps you transition from XGBoost to OpenBoost with minimal changes.

## Parameter Mapping

### XGBoost → OpenBoost

| XGBoost Parameter | OpenBoost Parameter | Notes |
|-------------------|---------------------|-------|
| `n_estimators` | `n_trees` / `n_estimators` | Same meaning |
| `max_depth` | `max_depth` | Same meaning |
| `learning_rate` / `eta` | `learning_rate` | Same meaning |
| `min_child_weight` | `min_child_weight` | Same meaning |
| `reg_lambda` / `lambda` | `reg_lambda` | L2 regularization |
| `reg_alpha` / `alpha` | `reg_alpha` | L1 regularization |
| `subsample` | `subsample` | Row sampling |
| `colsample_bytree` | `colsample_bytree` | Column sampling |
| `gamma` / `min_split_loss` | `gamma` | Min gain to split |
| `objective` | `loss` | See loss mapping below |

### Loss Function Mapping

| XGBoost Objective | OpenBoost Loss |
|-------------------|----------------|
| `reg:squarederror` | `'mse'` |
| `reg:absoluteerror` | `'mae'` |
| `reg:pseudohubererror` | `'huber'` |
| `binary:logistic` | `'logloss'` |
| `multi:softmax` | Use `MultiClassGradientBoosting` |
| `multi:softprob` | Use `MultiClassGradientBoosting` |
| `count:poisson` | `'poisson'` |
| `reg:gamma` | `'gamma'` |
| `reg:tweedie` | `'tweedie'` |

## Code Examples

### Basic Regression

```python
# XGBoost
import xgboost as xgb
model = xgb.XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    reg_lambda=1.0,
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

# OpenBoost equivalent
import openboost as ob
model = ob.GradientBoosting(
    n_trees=100,
    max_depth=6,
    learning_rate=0.1,
    reg_lambda=1.0,
    loss='mse',
)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

### Binary Classification

```python
# XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    objective='binary:logistic',
)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_test)[:, 1]

# OpenBoost equivalent
model = ob.GradientBoosting(
    n_trees=100,
    max_depth=6,
    loss='logloss',
)
model.fit(X_train, y_train)
logits = model.predict(X_test)
pred_proba = 1 / (1 + np.exp(-logits))  # Sigmoid

# Or use sklearn wrapper
from openboost import OpenBoostClassifier
model = OpenBoostClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_test)[:, 1]
```

### Multi-Class Classification

```python
# XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    objective='multi:softprob',
    num_class=5,
)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_test)
pred = model.predict(X_test)

# OpenBoost equivalent
model = ob.MultiClassGradientBoosting(
    n_classes=5,
    n_trees=100,
    max_depth=6,
)
model.fit(X_train, y_train)
pred_proba = model.predict_proba(X_test)
pred = model.predict(X_test)

# Or use sklearn wrapper
from openboost import OpenBoostClassifier
model = OpenBoostClassifier(n_estimators=100, max_depth=6)
model.fit(X_train, y_train)  # Auto-detects multi-class
```

### sklearn-Compatible API

OpenBoost provides drop-in replacements for XGBoost's sklearn API:

```python
# XGBoost sklearn
from xgboost import XGBRegressor, XGBClassifier

# OpenBoost sklearn (same interface!)
from openboost import OpenBoostRegressor, OpenBoostClassifier

# Works with sklearn pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', OpenBoostRegressor(n_estimators=100)),
])

# Works with cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(OpenBoostRegressor(), X, y, cv=5)

# Works with grid search
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(
    OpenBoostRegressor(),
    {'n_estimators': [50, 100], 'max_depth': [4, 6]},
    cv=3,
)
grid.fit(X, y)
```

### Early Stopping

```python
# XGBoost
model = xgb.XGBRegressor(
    n_estimators=1000,
    early_stopping_rounds=10,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)

# OpenBoost equivalent
from openboost import EarlyStopping, Logger

model = ob.GradientBoosting(n_trees=1000)
model.fit(
    X_train, y_train,
    callbacks=[
        EarlyStopping(patience=10),
        Logger(every=10),
    ],
    eval_set=[(X_val, y_val)],
)
```

### Feature Importance

```python
# XGBoost
model.fit(X_train, y_train)
importance = model.feature_importances_

# OpenBoost
model.fit(X_train, y_train)
importance = ob.compute_feature_importances(model.trees_)

# Or with sklearn wrapper
from openboost import OpenBoostRegressor
model = OpenBoostRegressor()
model.fit(X_train, y_train)
importance = model.feature_importances_  # Same as XGBoost!
```

### Saving and Loading

```python
# XGBoost
model.save_model('model.json')
loaded = xgb.XGBRegressor()
loaded.load_model('model.json')

# OpenBoost
model.save('model.joblib')
loaded = ob.GradientBoosting.load('model.joblib')

# Or with joblib directly (same as XGBoost pickle)
import joblib
joblib.dump(model, 'model.joblib')
loaded = joblib.load('model.joblib')
```

## Feature Comparison

| Feature | XGBoost | OpenBoost |
|---------|---------|-----------|
| GPU Support | ✅ | ✅ |
| Custom Loss | ⚠️ (requires Python wrapper) | ✅ (native Python) |
| Uncertainty | ❌ | ✅ (NaturalBoost) |
| Interpretable GAM | ❌ | ✅ (OpenBoostGAM) |
| Linear Leaves | ❌ | ✅ (LinearLeafGBDT) |
| DART | ✅ | ✅ |
| Growth Strategies | Level-wise | Level-wise, Leaf-wise, Symmetric |
| GOSS Sampling | ❌ (use LightGBM) | ✅ |
| Pure Python | ❌ (C++) | ✅ |

## What OpenBoost Does Better

### 1. Uncertainty Quantification

```python
# XGBoost: Just point predictions
pred = xgb_model.predict(X_test)  # Single number

# OpenBoost: Full distributions
model = ob.NaturalBoostNormal(n_trees=100)
model.fit(X_train, y_train)
mean = model.predict(X_test)
lower, upper = model.predict_interval(X_test)  # 90% interval
samples = model.sample(X_test, n_samples=1000)  # Monte Carlo
```

### 2. Custom Loss Functions

```python
# XGBoost: Requires Python callback wrapper, tricky to get right
# OpenBoost: Native Python, just return (grad, hess)
def my_loss(pred, y):
    grad = pred - y
    hess = np.ones_like(pred)
    return grad.astype(np.float32), hess.astype(np.float32)

model = ob.GradientBoosting(loss=my_loss)
```

### 3. Interpretable Models

```python
# XGBoost: SHAP values (post-hoc, expensive)
# OpenBoost: Inherently interpretable GAM
gam = ob.OpenBoostGAM(n_rounds=500)
gam.fit(X_train, y_train)
gam.plot_shape_function(0, feature_name="age")
```

### 4. Code Readability

```python
# XGBoost: 200K+ lines of C++
# OpenBoost: ~6K lines of Python you can actually read and modify
```

## What XGBoost Does Better

### 1. Raw Speed on CPU

XGBoost is highly optimized C++ and will be faster on CPU for very large datasets.

### 2. Distributed Training (Spark, Dask)

XGBoost has mature distributed training support.

### 3. Community and Ecosystem

XGBoost has more examples, tutorials, and community support.

## Migration Checklist

- [ ] Replace `xgb.XGBRegressor` with `ob.GradientBoosting` or `OpenBoostRegressor`
- [ ] Replace `xgb.XGBClassifier` with `ob.GradientBoosting(loss='logloss')` or `OpenBoostClassifier`
- [ ] Replace `n_estimators` with `n_trees` (or use sklearn wrapper)
- [ ] Replace `objective` with `loss`
- [ ] Update early stopping syntax
- [ ] Update feature importance code
- [ ] Update save/load code

## Gradual Migration

You can use both libraries during migration:

```python
import xgboost as xgb
import openboost as ob

# Keep XGBoost for existing models
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)

# Try OpenBoost for new features
ob_model = ob.NaturalBoostNormal()  # Uncertainty!
ob_model.fit(X_train, y_train)

# Compare predictions
xgb_pred = xgb_model.predict(X_test)
ob_pred = ob_model.predict(X_test)
print(f"Correlation: {np.corrcoef(xgb_pred, ob_pred)[0,1]:.4f}")
```

## Getting Help

- [Quickstart Guide](../quickstart.md) - Get started with OpenBoost
- [Uncertainty Tutorial](../tutorials/uncertainty.md) - Learn NaturalBoost
- [Custom Loss Tutorial](../tutorials/custom-loss.md) - Define your own objectives
