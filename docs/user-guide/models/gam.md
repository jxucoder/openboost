# OpenBoostGAM

GPU-accelerated Generalized Additive Model - interpretable machine learning with feature-level explanations.

## Why GAM?

GAMs decompose predictions into individual feature contributions:

```
prediction = f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ) + intercept
```

This means you can visualize exactly how each feature affects the prediction.

**Scope:** OpenBoostGAM learns *main effects only* — one shape function per
feature. Unlike InterpretML's EBM, it does not learn pairwise interaction
terms. If your signal depends heavily on feature interactions, EBM (with
interactions enabled) or a standard GBDT will fit better; OpenBoostGAM trades
that capacity for speed and a simpler, fully additive explanation.

## Basic Usage

```python
import openboost as ob

gam = ob.OpenBoostGAM(
    n_rounds=500,
    learning_rate=0.05,
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

`plot_shape_function(feature_idx, feature_name=None, ax=None)` returns the
matplotlib Axes it drew on; pass `ax=` to compose subplot grids.

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n_rounds` | int | 1000 | Number of boosting rounds |
| `learning_rate` | float | 0.01 | Step size |
| `reg_lambda` | float | 1.0 | L2 regularization on leaf values |
| `loss` | str/callable | 'mse' | Loss function (`'mse'`, `'logloss'`, or callable) |
| `n_bins` | int | 256 | Histogram bins (max 254 usable; bin 255 reserved for missing) |

`n_trees` is accepted as an alias for `n_rounds`. There is no `max_depth`
parameter: each round applies a regularized Newton update to every bin of
every feature's shape function (a per-feature lookup table), so smoothness is
controlled by `learning_rate`, `n_rounds`, and `reg_lambda` — not tree depth.

## Performance vs InterpretML EBM

One benchmark run is committed to the repo
(`benchmarks/results/gpu_benchmark_20260322_153105.json`, produced by
`benchmarks/compare_gpu.py --bench ebm` on a Modal A100):

| Dataset | OpenBoostGAM (GPU) | EBM (CPU) | Speedup | GAM R² | EBM R² |
|---------|-------------------|-----------|---------|--------|--------|
| Synthetic, 50,000 x 20 | 0.14s | 8.06s | **56x** | 0.663 | 0.738 |

Read this honestly: OpenBoostGAM was much faster **but less accurate** on
this run (R² 0.663 vs 0.738). Also note the comparison controls: both models
ran 200 rounds at `learning_rate=0.05`, and EBM was configured with
`interactions=0`, `outer_bags=1`, `inner_bags=0` — i.e. EBM's pairwise
interactions and bagging (both defaults that improve its accuracy but slow it
down) were disabled. With EBM defaults, expect the accuracy gap to widen and
the speed gap to grow.

Reproduce with:

```bash
uv run modal run benchmarks/compare_gpu.py --bench ebm
```

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

1. **Use more rounds** with lower learning rate for smoother shape functions
2. **Normalize features** for easier interpretation
3. **Check shape functions** for unexpected patterns (data issues)
4. **Compare against a standard GBDT** — if it beats the GAM by a wide
   margin, your data likely has interactions the GAM cannot capture
