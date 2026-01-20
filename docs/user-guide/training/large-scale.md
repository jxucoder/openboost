# Large-Scale Training

Train on datasets that don't fit in memory or need faster training.

## GOSS Sampling

Gradient-based One-Side Sampling (from LightGBM) - train 3x faster with minimal accuracy loss.

```python
import openboost as ob

model = ob.GradientBoosting(
    n_trees=100,
    subsample_strategy='goss',
    goss_top_rate=0.2,    # Keep top 20% high-gradient samples
    goss_other_rate=0.1,  # Sample 10% of the rest
)
model.fit(X_train, y_train)
```

### How GOSS Works

1. Sort samples by gradient magnitude
2. Keep top `goss_top_rate` samples (most informative)
3. Randomly sample `goss_other_rate` from the rest
4. Weight the random samples to maintain unbiased gradients

### GOSS Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `goss_top_rate` | 0.2 | Fraction of high-gradient samples to keep |
| `goss_other_rate` | 0.1 | Fraction of remaining samples to keep |

**Result**: Train on ~28% of samples with similar accuracy.

## Memory-Mapped Arrays

For datasets larger than RAM:

```python
import openboost as ob

# Create memory-mapped binned array (saves to disk)
X_mmap = ob.create_memmap_binned('large_data.npy', X_large)

# Load for training (no copy, uses disk)
X_mmap = ob.load_memmap_binned('large_data.npy', n_features, n_samples)

# Train as normal
model = ob.GradientBoosting(n_trees=100)
model.fit(X_mmap, y_train)
```

## Mini-Batch Training

Accumulate histograms in batches:

```python
from openboost import MiniBatchIterator, accumulate_histograms_minibatch

# Process 100k samples at a time
hist_grad, hist_hess = accumulate_histograms_minibatch(
    X_mmap, grad, hess,
    batch_size=100_000,
    n_features=n_features,
)
```

## Multi-GPU Training

Distribute training across multiple GPUs:

```python
import openboost as ob

# Automatic multi-GPU with Ray
model = ob.GradientBoosting(n_trees=100, n_gpus=4)
model.fit(X, y)

# Or specify exact devices
model = ob.GradientBoosting(n_trees=100, devices=[0, 2])
model.fit(X, y)
```

### Requirements

```bash
pip install "openboost[distributed]"  # Installs Ray
```

## Scaling Guidelines

| Dataset Size | Recommendation |
|--------------|----------------|
| <100K samples | Standard training |
| 100K-1M | GOSS sampling |
| 1M-10M | GOSS + memory-mapped |
| >10M | Multi-GPU + GOSS |

## Example: Large Dataset

```python
import numpy as np
import openboost as ob

# Simulate large dataset (10M samples)
n_samples = 10_000_000
n_features = 100

# Create memory-mapped data
X_mmap = ob.create_memmap_binned(
    'large_X.npy',
    np.random.randn(n_samples, n_features).astype(np.float32)
)

y = np.random.randn(n_samples).astype(np.float32)

# Train with GOSS
model = ob.GradientBoosting(
    n_trees=100,
    subsample_strategy='goss',
    goss_top_rate=0.1,
    goss_other_rate=0.05,
)
model.fit(X_mmap, y)
```
