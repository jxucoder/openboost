# OpenBoostGAM

GPU-accelerated Generalized Additive Model - interpretable machine learning with feature-level explanations.

## Why GAM?

GAMs decompose predictions into individual feature contributions:

```
prediction = f₁(x₁) + f₂(x₂) + ... + fₙ(xₙ) + intercept
```

This means you can visualize exactly how each feature affects the prediction.

**Scope:** By default OpenBoostGAM learns *main effects only* — one shape
function per feature. Setting `interactions=k` adds `k` pairwise interaction
terms (GA2M-style, like InterpretML's EBM), which closes part of the accuracy
gap to EBM while keeping every term inspectable. Be aware that the interaction
stage trains on CPU even when the main effects run on GPU. If your signal
depends heavily on higher-order interactions, a standard GBDT will still fit
better; OpenBoostGAM trades that capacity for speed and an explanation you can
plot term by term.

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
| `interactions` | int | 0 | Pairwise interaction terms (GA2M-style) learned after main effects; 0 = off |
| `interaction_rounds` | int/None | None | Boosting rounds for the interaction stage (None = `n_rounds`) |
| `smoothing` | float | 0.0 | Fused-ridge smoothing of 1D shape functions; 0 = off |
| `monotone` | dict/None | None | Per-feature monotonicity: `{feature_idx: +1}` (non-decreasing) or `-1` (non-increasing) |

All new parameters default to "off", preserving the exact pre-existing
behavior. `n_trees` is accepted as an alias for `n_rounds`. There is no
`max_depth` parameter: each round applies a regularized Newton update to every
bin of every feature's shape function (a per-feature lookup table), so
smoothness is controlled by `learning_rate`, `n_rounds`, `reg_lambda`, and
`smoothing` — not tree depth.

## Pairwise Interactions (GA2M)

With `interactions=k`, after main-effects training the model ranks all feature
pairs on the residual (FAST-style one-shot 2D Newton score, row-subsampled on
large datasets), picks the top `k`, and boosts a 2D shape table for each:

```python
gam = ob.OpenBoostGAM(n_rounds=500, learning_rate=0.05, interactions=3)
gam.fit(X_train, y_train)

gam.interaction_pairs_            # ranked list of (i, j) tuples
Z = gam.get_pair_shape_function(*gam.interaction_pairs_[0])  # (256, 256) table
```

`get_pair_shape_function(i, j)` returns a copy of the additive contribution on
the `(bin_i, bin_j)` grid and raises `KeyError` for pairs that were not
selected. Interaction terms are used automatically by `predict`.

**Honest scope note:** interactions close part of the gap to EBM (which enables
them by default) but the interaction stage always trains on CPU — expect the
GPU speedup story to apply to the main-effects stage only.

## Smoothing and Monotone Constraints

```python
gam = ob.OpenBoostGAM(
    n_rounds=500,
    learning_rate=0.05,
    smoothing=1.0,             # damp jagged, sparse-bin noise in shape functions
    monotone={0: +1, 3: -1},   # feature 0 non-decreasing, feature 3 non-increasing
)
gam.fit(X_train, y_train)
```

- `smoothing` solves a fused-ridge (first-difference penalty) system per round,
  so occupied bins anchor the shape while empty bins interpolate between their
  neighbors. It applies to ordinal (numeric) bins only — categorical features,
  the missing-value bin, and 2D interaction tables are not smoothed.
- `monotone` projects the accumulated shape function onto the constraint after
  every round with count-weighted isotonic regression (PAVA). Useful when
  domain knowledge says "risk never decreases with debt ratio".

## Validation, Callbacks, and Early Stopping

`fit` now accepts the same training-control arguments as `GradientBoosting`:

```python
gam = ob.OpenBoostGAM(n_rounds=2000, learning_rate=0.05)
gam.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[ob.Logger(period=100)],
    early_stopping_rounds=50,   # sugar for EarlyStopping(patience=50, restore_best=True)
)

gam.evals_result_     # {'eval_0': {'mse': [...]}} — per-round history per eval set
gam.best_iteration_   # set when early stopping is used
gam.best_score_
```

Early stopping monitors the **last** eval set; with `restore_best=True` (the
default via `early_stopping_rounds`) the shape tables are restored to the best
round. With `interactions > 0`, the main-effect and interaction rounds form
one monitored sequence — stopping during the main-effect phase skips the
interaction phase.

**Backend note:** smoothing, monotone projection, the interaction stage, and
the callbacks/eval_set machinery all run on CPU. On a CUDA backend, a plain
`fit()` keeps the GPU main-effects path; a fit that requests smoothing,
monotone constraints, callbacks, or eval_set falls back to CPU training with a
warning.

## Performance vs InterpretML EBM

One benchmark run is committed to the repo
(`benchmarks/results/gpu_benchmark_20260322_153105.json`, produced by
`benchmarks/compare_gpu.py --bench ebm` on a Modal A100):

| Dataset | OpenBoostGAM (GPU) | EBM (CPU) | Speedup | GAM R² | EBM R² |
|---------|-------------------|-----------|---------|--------|--------|
| Synthetic, 50,000 x 20 | 0.14s | 8.06s | **56x** | 0.663 | 0.738 |

Read this honestly: OpenBoostGAM was much faster **but less accurate** on
this run (R² 0.663 vs 0.738). Also note the comparison controls: both models
ran 200 rounds at `learning_rate=0.05`, and both were main-effects-only —
OpenBoostGAM predates its `interactions` support in this run, and EBM was
configured with `interactions=0`, `outer_bags=1`, `inner_bags=0` — i.e. EBM's
pairwise interactions and bagging (both defaults that improve its accuracy but
slow it down) were disabled. With EBM defaults, expect the accuracy gap to
widen and the speed gap to grow; enabling `interactions` on OpenBoostGAM
closes part of that gap but its interaction stage runs on CPU.

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
