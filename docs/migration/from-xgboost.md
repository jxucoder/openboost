# Migrating from XGBoost to OpenBoost

Stay on XGBoost or LightGBM for ordinary mean regression. They are
faster C++. Switch to OpenBoost for **distributional regression**
(`F(y | x)`), a **varying-coefficient** formula `y = f(θ(z), x)`, or a
Weibull AFT whose shape varies with covariates.

This page maps XGBoost APIs onto OpenBoost for the overlap, then shows the
three things XGBoost cannot express.

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
        Logger(period=10),
    ],
    eval_set=[(X_val, y_val)],
)
```

### Feature Importance

```python
# XGBoost
model.fit(X_train, y_train)
importance = model.feature_importances_

# OpenBoost (pass the fitted model)
model.fit(X_train, y_train)
importance = ob.compute_feature_importances(model)

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

## Feature comparison

| | XGBoost | OpenBoost |
|---|---|---|
| Point-estimate GBDT | Fast C++, the default choice | Works; not the reason to switch |
| GPU trees | Yes | Yes |
| Custom loss | Python `obj` callback; **diagonal Hessian only** | Native `(grad, hess)`; FormulaBoost does **full GGN** |
| Distributional / NGBoost-style | No | `NaturalBoost*` |
| Formula `y = f(θ, x)` with coupled params | Diagonal custom obj only | `FormulaBoost(precond="full")` |
| Survival AFT | Location only; scale is a **global** hyperparameter | `WeibullAFT` boosts `λ(z)` **and** `k(z)` |
| Interpretable GAM | No (use SHAP) | `OpenBoostGAM` |
| All Python | No | Yes (~20K lines) |

## Where OpenBoost adds something

### 1. A full distribution

```python
# XGBoost: a point
pred = xgb_model.predict(X_test)

# OpenBoost: parameters of a distribution
model = ob.NaturalBoostNormal(n_trees=100)
model.fit(X_train, y_train)
mean = model.predict(X_test)
lo, hi = model.predict_interval(X_test, alpha=0.1)
samples = model.sample(X_test, n_samples=1000)
```

On the UCI datasets measured so far, NLL is tied or better vs NGBoost, and on
an A100 NaturalBoost fits in seconds at sizes where CPU-only NGBoost needs
most of an hour. See [Benchmarks](../benchmarks.md) for the caveats.

### 2. A formula with off-diagonal GGN

XGBoost custom objectives cannot represent the off-diagonal of `JᵀJ`.
That term is what recovers coupled parameters (`b(z)` on the sales curve:
corr 0.877 vs 0.599). Black-box XGBoost also cannot extrapolate in the
structural input `x`. FormulaBoost is ~21x better there.

```python
def sales(theta, x):
    a, b = theta
    return a * x ** (1.0 / (1.0 + np.exp(-b * x)))

model = ob.FormulaBoost(
    formula=sales, n_params=2, links=("log", "identity"),
    param_names=("a", "b"), precond="full",
)
model.fit(Z, y, model_input=x)
params = model.predict_params(Z)   # the actual deliverable
```

### 3. Per-row Weibull shape

```python
# XGBoost AFT: one global scale hyperparameter, same k for every row
# OpenBoost: both λ(z) and k(z) are ensembles
model = ob.WeibullAFT(n_trees=300, max_depth=3)
model.fit(Z, time, event=observed)
params = model.predict_params(Z)           # {scale, shape}
s = model.predict_survival(Z, t=12.0)
```

On a varying-shape DGP, shape correlation is 0.997; XGBoost has no
per-row `k` to correlate. Censored NLL is better (0.761 vs 0.830);
C-index is close (0.680 vs 0.672).

### 4. A native Python custom loss (point-estimate)

```python
def my_loss(pred, y):
    grad = pred - y
    hess = np.ones_like(pred)
    return grad.astype(np.float32), hess.astype(np.float32)

model = ob.GradientBoosting(loss=my_loss)
```

## What XGBoost does better

- **Point-estimate speed on CPU.** Optimized C++. Use it for MSE/logloss.
- **Distributed training.** Spark / Dask / dedicated cluster runtimes.
- **Ecosystem.** More examples, more Stack Overflow, more production war stories.

If the job is "fit a GBDT, get a number," do not migrate.

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

# Keep XGBoost for existing point-estimate models
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# OpenBoost for F(y | x)
ob_model = ob.NaturalBoostNormal()
ob_model.fit(X_train, y_train)
ob_pred = ob_model.predict(X_test)
print(f"Correlation: {np.corrcoef(xgb_pred, ob_pred)[0, 1]:.4f}")
```

## Getting help

- [Quickstart](../getting-started/quickstart.md)
- [How it works](../user-guide/how-it-works.md)
- [FormulaBoost](../user-guide/formulaboost.md)
- [Weibull AFT](../user-guide/survival.md)
- [Uncertainty tutorial](../tutorials/uncertainty.md)
- [Benchmarks](../benchmarks.md)
