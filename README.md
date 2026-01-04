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
        ├── GradientBoosting    (standard GBDT)
        ├── OpenBoostGAM        (interpretable, EBM-style)
        └── YourAlgorithm       (build your own!)
```

**Built-in:**
- `GradientBoosting` — Standard GBDT with scikit-learn API
- `OpenBoostGAM` — GPU-accelerated interpretable model

**Easy to build:**
- Symmetric trees (CatBoost-style)
- DART (dropout trees)
- Custom research variants

### 🚀 Competitive Performance

Fast enough for production, while being infinitely more hackable than C++ alternatives.

- Competitive with XGBoost at medium scale
- GPU-GAM dramatically faster than CPU-based alternatives
- The tradeoff: We optimize for flexibility, not just raw speed

## Who Is OpenBoost For?

### 🏆 Kaggle Competitors

Test custom losses in minutes, not hours:

```python
def focal_loss(pred, y, gamma=2.0):
    p = sigmoid(pred)
    grad = p - y  # simplified
    hess = p * (1 - p)
    return grad, hess

model = ob.GradientBoosting(loss=focal_loss)
model.fit(X, y)
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

- [x] Standard GBDT (`GradientBoosting`)
- [x] GPU-accelerated GAM (`OpenBoostGAM`)
- [ ] Symmetric/Oblivious trees
- [ ] DART (dropout trees)
- [ ] Multi-GPU support
- [ ] More tree variants (contributions welcome!)

## License

MIT
