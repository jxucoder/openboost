# sklearn Integration

OpenBoost provides sklearn-compatible wrappers for seamless integration with scikit-learn pipelines.

## Available Wrappers

| Wrapper | Base Model | Use Case |
|---------|------------|----------|
| `OpenBoostRegressor` | GradientBoosting | Regression |
| `OpenBoostClassifier` | GradientBoosting | Classification |
| `OpenBoostDistributionalRegressor` | NaturalBoost | Probabilistic regression |
| `OpenBoostLinearLeafRegressor` | LinearLeafGBDT | Linear leaf regression |

## Basic Usage

```python
from openboost import OpenBoostRegressor, OpenBoostClassifier

# Regressor
reg = OpenBoostRegressor(n_estimators=100, max_depth=6)
reg.fit(X_train, y_train)
print(f"R² Score: {reg.score(X_test, y_test):.4f}")

# Classifier
clf = OpenBoostClassifier(n_estimators=100, max_depth=6)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")
```

## Cross-Validation

```python
from sklearn.model_selection import cross_val_score

reg = OpenBoostRegressor(n_estimators=100)
scores = cross_val_score(reg, X, y, cv=5)
print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")
```

## Grid Search

```python
from sklearn.model_selection import GridSearchCV
from openboost import OpenBoostRegressor, get_param_grid

# Get suggested parameter grid
param_grid = get_param_grid('regression')

# Or define your own
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
}

search = GridSearchCV(OpenBoostRegressor(), param_grid, cv=5)
search.fit(X, y)
print(f"Best params: {search.best_params_}")
```

## Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from openboost import OpenBoostRegressor

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', OpenBoostRegressor(n_estimators=100)),
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Parameter Mapping

sklearn wrapper parameters map to OpenBoost parameters:

| sklearn Parameter | OpenBoost Parameter |
|-------------------|---------------------|
| `n_estimators` | `n_trees` |
| `max_depth` | `max_depth` |
| `learning_rate` | `learning_rate` |
| `min_samples_leaf` | `min_child_weight` |
| `subsample` | `subsample` |

## Out-of-Fold Predictions

```python
from openboost import cross_val_predict, cross_val_predict_proba

# Regression
oof_pred = cross_val_predict(model, X, y, cv=5)

# Classification
oof_proba = cross_val_predict_proba(classifier, X, y, cv=5)
```

## Distributional sklearn Wrapper

```python
from openboost import OpenBoostDistributionalRegressor

model = OpenBoostDistributionalRegressor(
    distribution='normal',
    n_estimators=100,
)
model.fit(X_train, y_train)

# Get prediction intervals
lower, upper = model.predict_interval(X_test, alpha=0.1)
```
