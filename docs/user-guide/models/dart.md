# DART

Dropout Additive Regression Trees - a regularization technique that randomly drops trees during training.

## Why DART?

Standard gradient boosting can overfit, especially with many trees. DART addresses this by:

1. Randomly dropping trees during each boosting iteration
2. Normalizing the contribution of remaining trees
3. Preventing individual trees from dominating

## Basic Usage

```python
import openboost as ob

model = ob.DART(
    n_trees=200,
    max_depth=6,
    dropout_rate=0.1,  # Drop 10% of trees
)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dropout_rate` | float | 0.1 | Fraction of trees to drop |
| `skip_drop` | float | 0.5 | Probability of skipping dropout |

Plus all standard `GradientBoosting` parameters.

## When to Use DART

| Situation | Recommendation |
|-----------|----------------|
| Overfitting with many trees | Try DART |
| Small dataset | DART can help |
| Large dataset | Standard GBDT usually fine |
| Need interpretability | Standard GBDT (DART trees interact) |

## Example

```python
import openboost as ob
from sklearn.model_selection import cross_val_score

# Compare standard GBDT vs DART
gbdt = ob.GradientBoosting(n_trees=200, max_depth=6)
dart = ob.DART(n_trees=200, max_depth=6, dropout_rate=0.1)

gbdt_scores = cross_val_score(gbdt, X, y, cv=5)
dart_scores = cross_val_score(dart, X, y, cv=5)

print(f"GBDT: {gbdt_scores.mean():.4f} ± {gbdt_scores.std():.4f}")
print(f"DART: {dart_scores.mean():.4f} ± {dart_scores.std():.4f}")
```
