# Multi-class Classification

For classification problems with more than 2 classes.

## Basic Usage

```python
import openboost as ob

model = ob.MultiClassGradientBoosting(
    n_classes=5,
    n_trees=100,
    max_depth=6,
)
model.fit(X_train, y_train)  # y_train: 0, 1, 2, 3, or 4

# Get class probabilities
probabilities = model.predict_proba(X_test)  # Shape: (n_samples, n_classes)

# Get class predictions
predictions = model.predict(X_test)  # Shape: (n_samples,)
```

## Parameters

Same as `GradientBoosting`, plus:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_classes` | int | required | Number of classes |

## Example: Iris Classification

```python
import numpy as np
import openboost as ob
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X, y = iris.data.astype(np.float32), iris.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train
model = ob.MultiClassGradientBoosting(
    n_classes=3,
    n_trees=100,
    max_depth=4,
)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2%}")
```

## sklearn Wrapper

For scikit-learn compatibility:

```python
from openboost import OpenBoostClassifier
from sklearn.model_selection import cross_val_score

clf = OpenBoostClassifier(n_estimators=100, max_depth=6)
scores = cross_val_score(clf, X, y, cv=5)
print(f"CV Accuracy: {scores.mean():.2%}")
```
