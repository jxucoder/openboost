# Quickstart

Get up and running with OpenBoost in 5 minutes.

## Basic Regression

```python
import numpy as np
import openboost as ob

# Generate sample data
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] * 2 + X[:, 1] + np.random.randn(1000) * 0.1).astype(np.float32)

# Split data
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

# Train model
model = ob.GradientBoosting(
    n_trees=100,
    max_depth=6,
    learning_rate=0.1,
    loss='mse',
)
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
rmse = np.sqrt(np.mean((predictions - y_test) ** 2))
print(f"RMSE: {rmse:.4f}")
```

## Binary Classification

```python
import openboost as ob

model = ob.GradientBoosting(
    n_trees=100,
    max_depth=6,
    loss='logloss',
)
model.fit(X_train, y_train)  # y_train: 0 or 1

# Get probabilities
logits = model.predict(X_test)
probabilities = 1 / (1 + np.exp(-logits))

# Get class predictions
predictions = (probabilities > 0.5).astype(int)
```

## Multi-Class Classification

```python
import openboost as ob

model = ob.MultiClassGradientBoosting(
    n_classes=5,
    n_trees=100,
    max_depth=6,
)
model.fit(X_train, y_train)  # y_train: 0, 1, 2, 3, or 4

# Get probabilities
probabilities = model.predict_proba(X_test)  # Shape: (n_samples, n_classes)

# Get class predictions
predictions = model.predict(X_test)
```

## Uncertainty Quantification

```python
import openboost as ob

# Train probabilistic model
model = ob.NaturalBoostNormal(n_trees=100, max_depth=4)
model.fit(X_train, y_train)

# Point prediction (mean)
mean = model.predict(X_test)

# 90% prediction interval
lower, upper = model.predict_interval(X_test, alpha=0.1)

# Sample from predicted distribution
samples = model.sample(X_test, n_samples=100)
```

## sklearn-Compatible API

```python
from openboost import OpenBoostRegressor, OpenBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Regressor
reg = OpenBoostRegressor(n_estimators=100, max_depth=6)
reg.fit(X_train, y_train)
print(f"R² Score: {reg.score(X_test, y_test):.4f}")

# Cross-validation
scores = cross_val_score(reg, X, y, cv=5)
print(f"CV Score: {scores.mean():.4f} ± {scores.std():.4f}")

# Grid search
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.05, 0.1, 0.2],
}
grid = GridSearchCV(reg, param_grid, cv=3)
grid.fit(X_train, y_train)
print(f"Best params: {grid.best_params_}")
```

## Callbacks

```python
import openboost as ob
from openboost import EarlyStopping, Logger

model = ob.GradientBoosting(n_trees=500, max_depth=6)

callbacks = [
    EarlyStopping(patience=10, min_delta=0.001),
    Logger(every=10),
]

model.fit(
    X_train, y_train,
    callbacks=callbacks,
    eval_set=[(X_test, y_test)],
)

print(f"Stopped at {len(model.trees_)} trees")
```

## Saving and Loading

```python
import openboost as ob

# Train
model = ob.GradientBoosting(n_trees=100)
model.fit(X_train, y_train)

# Save
model.save('my_model.joblib')

# Load
loaded_model = ob.GradientBoosting.load('my_model.joblib')
predictions = loaded_model.predict(X_test)
```

## Next Steps

- [GPU Setup](gpu-setup.md) - Configure GPU acceleration
- [Uncertainty Quantification](../tutorials/uncertainty.md) - Deep dive into NaturalBoost
- [Custom Loss Functions](../tutorials/custom-loss.md) - Define your own objectives
