# OpenBoostGAM

GPU-accelerated Generalized Additive Model - interpretable machine learning with feature-level explanations.

## Why GAM?

GAMs decompose predictions into individual feature contributions:

```
prediction = f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ) + intercept
```

This means you can visualize exactly how each feature affects the prediction.

## Basic Usage

```python
import openboost as ob

gam = ob.OpenBoostGAM(
    n_rounds=500,
    learning_rate=0.05,
    max_depth=3,  # Shallow trees for smoothness
)
gam.fit(X_train, y_train)
predictions = gam.predict(X_test)

# Get feature importance
importance = gam.get_feature_importance()
```

## Visualizing Shape Functions

```python
# Plot how each feature affects predictions
gam.plot_shape_function(
    feature_idx=0,
    feature_name="age",
)

# Plot multiple features
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
for i, ax in enumerate(axes.flat):
    gam.plot_shape_function(i, ax=ax, feature_name=feature_names[i])
plt.tight_layout()
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_rounds` | int | 1000 | Number of boosting rounds |
| `learning_rate` | float | 0.01 | Step size |
| `max_depth` | int | 3 | Tree depth (keep small for smoothness) |
| `loss` | str | 'mse' | Loss function |

## Performance vs InterpretML EBM

| Samples | EBM (CPU) | OpenBoostGAM (GPU) | Speedup |
|---------|-----------|-------------------|---------|
| 10,000 | 3.6s | 0.14s | **25x** |
| 50,000 | 6.3s | 0.16s | **39x** |
| 100,000 | 10.5s | 0.25s | **43x** |

## Example: Credit Risk

```python
import openboost as ob
import matplotlib.pyplot as plt

# Train interpretable model
gam = ob.OpenBoostGAM(n_rounds=500, learning_rate=0.05)
gam.fit(X_train, y_train)

# Explain predictions
feature_names = ['age', 'income', 'debt_ratio', 'credit_history']

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for i, (ax, name) in enumerate(zip(axes.flat, feature_names)):
    gam.plot_shape_function(i, ax=ax, feature_name=name)
    ax.set_title(f"Effect of {name}")
plt.tight_layout()
plt.savefig("gam_explanations.png")
```

## Best Practices

1. **Use shallow trees** (`max_depth=3`) for smoother shape functions
2. **Use more rounds** with lower learning rate for better fits
3. **Normalize features** for easier interpretation
4. **Check shape functions** for unexpected patterns (data issues)
