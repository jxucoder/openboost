# Project 0: MVP with sklearn Interface

## Goal

Ship the simplest useful GPU gradient boosting in Python.

```python
import openboost as ob

model = ob.GradientBoosting(n_trees=100, max_depth=6)
model.fit(X_train, y_train)
pred = model.predict(X_test)
```

## Constraints

| Constraint | Value |
|------------|-------|
| Platform | Linux/Windows + NVIDIA GPU |
| No Mac | numba-cuda doesn't support it |
| Loss | MSE only |
| Data | Dense float32 arrays |

## Public API

```python
class GradientBoosting:
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ): ...
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting": ...
    
    def predict(self, X: np.ndarray) -> np.ndarray: ...
```

## Dependencies

```toml
dependencies = [
    "numpy>=1.24",
    "numba-cuda>=0.23",
]
```

## File Structure

```
src/openboost/
├── __init__.py       # exports GradientBoosting
├── _boosting.py      # GradientBoosting class
├── _array.py         # quantile binning to uint8
├── _tree.py          # Tree class, fit_tree()
├── _histogram.py     # histogram kernel dispatch
├── _split.py         # find best split
└── _kernels.py       # all @cuda.jit kernels
```

## Key Implementation Notes

### 1. Kernels at Module Level

**Critical**: Never define `@cuda.jit` inside functions.

### 2. Binning

Quantile binning to uint8 (256 bins max). Feature-major layout `(n_features, n_samples)` for coalesced GPU access.

### 3. Data Structures (enable future optimizations)

Use `sample_node_ids` array — enables level-order tree building later:

```python
# sample_node_ids[i] = which node sample i belongs to
sample_node_ids = np.zeros(n_samples, dtype=np.int32)  # all start at root
```

Store tree as flat arrays (not recursive objects) — enables GPU-native prediction:

```python
feature_idx: np.ndarray   # (max_nodes,)
threshold: np.ndarray     # (max_nodes,)
left_child: np.ndarray    # (max_nodes,)
right_child: np.ndarray   # (max_nodes,)
value: np.ndarray         # (max_nodes,)
```

## Success Criteria

```python
import openboost as ob
import numpy as np

# Generate data
X = np.random.randn(100_000, 50).astype(np.float32)
y = np.random.randn(100_000).astype(np.float32)

# Train
model = ob.GradientBoosting(n_trees=20, max_depth=6)
model.fit(X, y)

# Predict
pred = model.predict(X)

# Should:
# 1. Run without errors
# 2. Complete in < 5 seconds
# 3. Produce reasonable MSE (< 1.0 for this synthetic data)
```

## Results

**Status: ✅ Complete**

Tested on NVIDIA T4 GPU via Modal (CUDA 12.4, Python 3.11).

### Benchmark vs XGBoost

| Metric | OpenBoost | XGBoost GPU | Ratio |
|--------|-----------|-------------|-------|
| Train time | 4.88s | 1.36s | 3.6x |
| Predict time | 0.38s | 0.11s | 3.5x |
| MSE | 0.9938 | 0.9744 | ~same |

Config: `n_trees=20, max_depth=6, learning_rate=0.1` on 100K samples × 50 features.

### Success Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Run without errors | ✓ | ✓ | ✅ |
| Training < 5s | < 5s | 4.88s | ✅ |
| MSE < 1.0 | < 1.0 | 0.9938 | ✅ |

### Notes

- 3-4x slower than XGBoost is expected for pure Python/Numba implementation
- MSE is comparable, confirming correctness
- First run includes JIT compilation overhead; subsequent runs faster
- Low grid size warnings at early tree levels (1, 2, 4 nodes) are expected

## Dev Workflow

Since development is on Mac (no CUDA), test via Modal:

```bash
# Write code locally
# Test on remote GPU
modal run tests/test_modal.py
```

## What's NOT in Project 0

- CPU fallback
- Custom loss functions
- Early stopping
- Feature importance
- Serialization (save/load)
- Multi-output
- Categorical features
- Missing value handling
