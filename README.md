# OpenBoost

> **The GPU-native, all-Python platform for tree-based machine learning.**

Build any tree-based algorithm in Python, run it on GPU.

## Why OpenBoost?

**XGBoost and LightGBM are products. OpenBoost is a platform.**

| | XGBoost / LightGBM | OpenBoost |
|---|---|---|
| **Code** | 200K+ lines of C++ | ~6K lines of Python |
| **GPU** | Added later (bolt-on) | Native (designed for it) |
| **Customize** | Write C++, recompile | Modify Python, reload |
| **New algorithms** | Wait for maintainers | Build it yourself |

## Quick Start

### High-Level API

```python
import openboost as ob

# Standard gradient boosting
model = ob.GradientBoosting(n_trees=100, max_depth=6, loss='mse')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Probabilistic predictions with uncertainty (like NGBoost, but faster!)
prob_model = ob.NaturalBoostNormal(n_trees=100)
prob_model.fit(X_train, y_train)
mean = prob_model.predict(X_test)
lower, upper = prob_model.predict_interval(X_test, alpha=0.1)  # 90% interval

# Interpretable GAM (like Microsoft's EBM, but GPU-accelerated)
gam = ob.OpenBoostGAM(n_rounds=500, learning_rate=0.05)
gam.fit(X_train, y_train)
gam.plot_shape_function(0, feature_name="age")  # Visualize feature effects
```

### Low-Level API (Full Control)

```python
import openboost as ob

# Bin data once, reuse everywhere
X_binned = ob.array(X_train)

# You own the training loop
pred = np.zeros(len(y_train), dtype=np.float32)

for round in range(100):
    # Your loss, your gradients
    grad = 2 * (pred - y_train)  # MSE gradient
    hess = np.ones_like(grad) * 2
    
    # We build the tree (GPU-accelerated)
    tree = ob.fit_tree(X_binned, grad, hess, max_depth=6)
    pred = pred + 0.1 * tree(X_binned)
```

### PyTorch/JAX Integration

```python
import openboost as ob
import torch

# Zero-copy gradient flow from PyTorch to tree building
pred = torch.zeros(n_samples, device="cuda", requires_grad=True)
loss = my_custom_loss(pred, y)
grad = torch.autograd.grad(loss, pred)[0]

tree = ob.fit_tree(X_binned, grad, hess)  # No data copy!
```

## Installation

```bash
# With uv (recommended)
uv add openboost

# With GPU support
uv add "openboost[cuda]"

# Or with pip
pip install openboost
pip install openboost[cuda]
```

## The Four Pillars

### 🐍 All Python

Every kernel, every algorithm — readable Python you can understand and modify.

```python
# This is actual OpenBoost code. You can read it.
@cuda.jit
def _histogram_kernel(binned, grad, hess, hist_grad, hist_hess):
    feature_idx = cuda.blockIdx.x
    local_grad = cuda.shared.array(256, dtype=float32)
    # ... the rest is just as readable
```

### ⚡ GPU Native

Designed for GPU from day one. Not a CPU algorithm ported to GPU.

- Numba CUDA kernels optimized for modern GPUs
- GPU-first data structures and memory layout  
- CPU fallback for development (works on Mac)

### 🔧 Flexible Architecture

The same building blocks power different algorithms:

```
OpenBoost Core (Histograms, Binning, Loss Functions)
        │
        ├── GradientBoosting      (standard GBDT)
        ├── NaturalBoost          (probabilistic, NGBoost-style)
        ├── OpenBoostGAM          (interpretable, EBM-style)
        ├── LinearLeafGBDT        (linear models in leaves)
        ├── DART                  (dropout regularization)
        └── YourAlgorithm         (build your own!)
```

## Supported Models

### Standard GBDT
| Model | Description | Use Case |
|-------|-------------|----------|
| `GradientBoosting` | Standard gradient boosting | General regression/classification |
| `MultiClassGradientBoosting` | Multi-class with softmax | Multi-class classification |
| `DART` | Dropout regularized trees | Reduce overfitting |
| `OpenBoostGAM` | Interpretable additive model | Explainable ML |
| `LinearLeafGBDT` | Linear models in leaves | Better extrapolation |

### Probabilistic GBDT (NaturalBoost)
Full probability distributions, not just point estimates!

| Model | Distribution | Use Case |
|-------|--------------|----------|
| `NaturalBoostNormal` | Gaussian | General uncertainty |
| `NaturalBoostLogNormal` | Log-Normal | Positive skewed (prices) |
| `NaturalBoostGamma` | Gamma | Positive continuous |
| `NaturalBoostPoisson` | Poisson | Count data |
| `NaturalBoostStudentT` | Student-t | Heavy tails, outliers |
| `NaturalBoostTweedie` | Tweedie | Insurance claims (Kaggle!) |
| `NaturalBoostNegBin` | Negative Binomial | Sales forecasting (Kaggle!) |
| `NaturalBoost(distribution=custom)` | Any custom | Define your own! |

```python
# Probabilistic predictions with uncertainty
model = ob.NaturalBoostNormal(n_trees=100)
model.fit(X_train, y_train)

mean = model.predict(X_test)                    # Point estimate
lower, upper = model.predict_interval(X_test)   # 90% confidence interval
samples = model.sample(X_test, n_samples=1000)  # Monte Carlo samples
```

### Growth Strategies
| Strategy | Style | Best For |
|----------|-------|----------|
| Level-wise | XGBoost | Balanced trees |
| Leaf-wise | LightGBM | Deep trees, large data |
| Symmetric | CatBoost | Fast inference |

### Large-Scale Training (Phase 17)

**GOSS (Gradient-based One-Side Sampling)** - Train 3x faster with minimal accuracy loss:

```python
# Keep top 20% of high-gradient samples, subsample 10% of the rest
# Total: ~28% of samples, but maintains accuracy!
model = ob.GradientBoosting(
    n_trees=100,
    subsample_strategy='goss',
    goss_top_rate=0.2,
    goss_other_rate=0.1,
)
model.fit(X_train, y_train)

# sklearn wrapper supports it too
from openboost import OpenBoostRegressor
model = OpenBoostRegressor(subsample_strategy='goss')
model.fit(X, y)
```

**Memory-Mapped Arrays** - Train on datasets larger than RAM:

```python
import openboost as ob

# Create memory-mapped binned array (saves to disk)
X_mmap = ob.create_memmap_binned('large_data.npy', X_large)

# Load for training (no copy, uses disk)
X_mmap = ob.load_memmap_binned('large_data.npy', n_features, n_samples)

# Mini-batch histogram accumulation
from openboost import MiniBatchIterator, accumulate_histograms_minibatch

# Process 100k samples at a time
hist_grad, hist_hess = accumulate_histograms_minibatch(
    X_mmap, grad, hess,
    batch_size=100_000,
    n_features=n_features,
)
```

### Loss Functions
| Loss | Function | Use Case |
|------|----------|----------|
| MSE | `mse_gradient` | Regression |
| MAE | `mae_gradient` | Robust regression |
| Huber | `huber_gradient` | Outlier-robust |
| Quantile | `quantile_gradient` | Quantile regression |
| LogLoss | `logloss_gradient` | Binary classification |
| Softmax | `softmax_gradient` | Multi-class |
| Poisson | `poisson_gradient` | Count data |
| Gamma | `gamma_gradient` | Positive continuous |
| Tweedie | `tweedie_gradient` | Zero-inflated positive |
| Custom | `callable(pred, y) -> (grad, hess)` | Anything! |

### 🚀 Competitive Performance

Fast enough for production, while being infinitely more hackable than C++ alternatives.

- Competitive with XGBoost at medium scale
- GPU-GAM dramatically faster than CPU-based alternatives
- NaturalBoost **1.3x faster** than NGBoost at 50k+ samples
- The tradeoff: We optimize for flexibility, not just raw speed

#### NaturalBoost vs NGBoost

| Feature | NGBoost | NaturalBoost |
|---------|---------|--------------|
| GPU Support | ❌ CPU only | ✅ GPU accelerated |
| Speed (50k samples) | 13.0s | **9.5s (1.4x faster)** |
| Custom distributions | Limited | ✅ Any parametric form |
| Autodiff gradients | ❌ | ✅ JAX or numerical |

## Who Is OpenBoost For?

### 🏆 Kaggle Competitors

**Probabilistic predictions** that XGBoost can't do:

```python
# Insurance claims (Porto Seguro, Allstate)
model = ob.NaturalBoostTweedie(power=1.5, n_trees=500)
model.fit(X_train, y_train)

# XGBoost: predict(X) → single number
# NaturalBoost: full distribution!
mean = model.predict(X_test)
lower, upper = model.predict_interval(X_test)  # Uncertainty bounds
samples = model.sample(X_test, n_samples=1000)  # Monte Carlo

# Sales forecasting (Rossmann, Bike Sharing)
model = ob.NaturalBoostNegBin(n_trees=500)  # Overdispersed counts
model.fit(X_train, y_train)

# Probability of high demand (inventory planning!)
output = model.predict_distribution(X_test)
prob_high = output.distribution.prob_exceed(output.params, threshold=100)
```

### 🔬 ML Researchers

Prototype new algorithms in Python:

```python
# Implement your paper's contribution
# No C++ required, no waiting for library maintainers
```

### 🏗️ Startups & Prototypes

Ship interpretable models fast:

```python
gam = ob.OpenBoostGAM(n_rounds=500)
gam.fit(X, y)

# Show stakeholders exactly how each feature affects predictions
gam.plot_shape_function(feature_idx=0, name="customer_age")
```

### 📚 Students & Educators

Actually understand how gradient boosting works:

```python
# Read the source. It's Python. It's ~6000 lines total.
# _histogram.py, _tree.py, _gam.py — all documented
```

## API Reference

### Data

```python
X_binned = ob.array(X, n_bins=256)  # Bin features (done once)
```

### High-Level Training

```python
# Standard GBDT
model = ob.GradientBoosting(
    n_trees=100,
    max_depth=6,
    learning_rate=0.1,
    loss='mse',  # or 'logloss', 'huber', or custom callable
)
model.fit(X, y)
pred = model.predict(X_test)

# Interpretable GAM
gam = ob.OpenBoostGAM(
    n_rounds=1000,
    learning_rate=0.01,
    loss='mse',
)
gam.fit(X, y)
pred = gam.predict(X_test)
importance = gam.get_feature_importance()
```

### Low-Level Training

```python
tree = ob.fit_tree(
    X_binned,
    grad,               # (n_samples,) float32
    hess,               # (n_samples,) float32
    max_depth=6,
    min_child_weight=1.0,
    reg_lambda=1.0,
)
predictions = tree(X_binned)
```

## Development

```bash
# Clone
git clone https://github.com/your-org/openboost.git
cd openboost

# Install with dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Run GPU benchmarks on Modal
uv run modal run benchmarks/modal_bench.py
```

## Philosophy

- **All Python**: Readable, hackable, no C++ required
- **GPU Native**: Designed for GPU from day one
- **Flexible**: Build any tree-based algorithm
- **Explicit**: You see every gradient, every tree, every decision

## Roadmap

### Completed ✅
- [x] Standard GBDT (`GradientBoosting`)
- [x] GPU-accelerated GAM (`OpenBoostGAM`)
- [x] DART (dropout trees)
- [x] Growth strategies (level-wise, leaf-wise, symmetric)
- [x] Loss functions (MSE, MAE, Huber, Quantile, LogLoss, Softmax, Poisson, Gamma, Tweedie)
- [x] Multi-class classification (`MultiClassGradientBoosting`)
- [x] **NaturalBoost** - Probabilistic GBDT with uncertainty quantification
- [x] **LinearLeafGBDT** - Trees with linear models in leaves
- [x] **Custom distributions** - Define any parametric form with autodiff
- [x] sklearn-compatible wrappers
- [x] Callbacks (early stopping, checkpointing, LR scheduling)
- [x] Feature importance (gain, split count, permutation)
- [x] **GOSS** - Gradient-based One-Side Sampling (LightGBM-style)
- [x] **Mini-batch training** - Handle datasets larger than memory
- [x] Memory-mapped array support

### In Progress 🚧
- [ ] Multi-GPU support (data parallelism)
- [ ] External memory (out-of-core training)

### Future 🔮
- [ ] More tree variants (contributions welcome!)

## License

Apache 2.0
