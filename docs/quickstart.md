# Quickstart Guide

Get up and running with OpenBoost in 5 minutes.

## Installation

```bash
# With uv (recommended)
uv add openboost

# With pip
pip install openboost

# With GPU support
uv add "openboost[cuda]"
pip install "openboost[cuda]"
```

## Basic Usage

### Regression

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

### Binary Classification

```python
import openboost as ob

# Train binary classifier
model = ob.GradientBoosting(
    n_trees=100,
    max_depth=6,
    loss='logloss',
)
model.fit(X_train, y_train)  # y_train: 0 or 1

# Get raw predictions (logits)
logits = model.predict(X_test)

# Convert to probabilities
probabilities = 1 / (1 + np.exp(-logits))

# Get class predictions
predictions = (probabilities > 0.5).astype(int)
```

### Multi-Class Classification

```python
import openboost as ob

# Train multi-class classifier
model = ob.MultiClassGradientBoosting(
    n_classes=5,  # Number of classes
    n_trees=100,
    max_depth=6,
)
model.fit(X_train, y_train)  # y_train: 0, 1, 2, 3, or 4

# Get class probabilities
probabilities = model.predict_proba(X_test)  # Shape: (n_samples, n_classes)

# Get class predictions
predictions = model.predict(X_test)  # Shape: (n_samples,)
```

## sklearn-Compatible API

OpenBoost provides sklearn-compatible wrappers for easy integration:

```python
from openboost import OpenBoostRegressor, OpenBoostClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV

# Regressor
reg = OpenBoostRegressor(n_estimators=100, max_depth=6)
reg.fit(X_train, y_train)
print(f"R² Score: {reg.score(X_test, y_test):.4f}")

# Classifier
clf = OpenBoostClassifier(n_estimators=100, max_depth=6)
clf.fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test):.4f}")

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

## Uncertainty Quantification

OpenBoost's NaturalBoost provides probabilistic predictions:

```python
import openboost as ob

# Train probabilistic model
model = ob.NaturalBoostNormal(n_trees=100, max_depth=4)
model.fit(X_train, y_train)

# Point prediction (mean)
mean = model.predict(X_test)

# 90% prediction interval
lower, upper = model.predict_interval(X_test, alpha=0.1)

# Full distribution output
output = model.predict_distribution(X_test)
print(f"Mean: {output.mean()[:5]}")
print(f"Std:  {output.std()[:5]}")

# Sample from predicted distribution
samples = model.sample(X_test, n_samples=100)  # Shape: (n_samples, n_test)
```

## Saving and Loading Models

```python
import openboost as ob

# Train
model = ob.GradientBoosting(n_trees=100)
model.fit(X_train, y_train)

# Save
model.save('my_model.joblib')

# Load
loaded_model = ob.GradientBoosting.load('my_model.joblib')

# Predictions match
np.testing.assert_allclose(
    model.predict(X_test),
    loaded_model.predict(X_test)
)

# Also works with pickle/joblib directly
import joblib
joblib.dump(model, 'model.joblib')
loaded = joblib.load('model.joblib')
```

## Callbacks for Training Control

```python
import openboost as ob
from openboost import EarlyStopping, Logger

# Early stopping
model = ob.GradientBoosting(n_trees=500, max_depth=6)

callbacks = [
    EarlyStopping(patience=10, min_delta=0.001),
    Logger(every=10),  # Print progress every 10 trees
]

model.fit(
    X_train, y_train,
    callbacks=callbacks,
    eval_set=[(X_test, y_test)],
)

print(f"Stopped at {len(model.trees_)} trees")
```

## Feature Importance

```python
import openboost as ob

model = ob.GradientBoosting(n_trees=100)
model.fit(X_train, y_train)

# Compute importance (gain-based)
importance = ob.compute_feature_importances(model.trees_)
print("Feature importances:", importance)

# Get as dictionary with feature names
feature_names = ['age', 'income', 'score', ...]
importance_dict = ob.get_feature_importance_dict(model.trees_, feature_names)

# Plot (requires matplotlib)
ob.plot_feature_importances(model.trees_, feature_names)
```

## Large-Scale Training

For datasets that don't fit in memory or need faster training:

```python
import openboost as ob

# GOSS sampling (3x faster with minimal accuracy loss)
model = ob.GradientBoosting(
    n_trees=100,
    subsample_strategy='goss',
    goss_top_rate=0.2,    # Keep top 20% high-gradient samples
    goss_other_rate=0.1,  # Sample 10% of the rest
)
model.fit(X_train, y_train)

# Memory-mapped arrays for out-of-core training
X_mmap = ob.create_memmap_binned('data.npy', X_large)
X_mmap = ob.load_memmap_binned('data.npy', n_features, n_samples)
```

## Next Steps

- [Uncertainty Quantification Tutorial](tutorials/uncertainty.md) - Deep dive into NaturalBoost
- [Custom Loss Functions](tutorials/custom-loss.md) - Define your own loss functions
- [Migration from XGBoost](migration/from-xgboost.md) - Switching from XGBoost to OpenBoost
