# GPU Setup

OpenBoost automatically detects and uses CUDA GPUs when available.

## Verify GPU Detection

```python
import openboost as ob

print(f"Backend: {ob.get_backend()}")  # "cuda" or "cpu"
print(f"Using GPU: {ob.is_cuda()}")    # True if GPU active
```

## Manual Backend Selection

```python
import openboost as ob

# Force CPU (useful for debugging or comparison)
ob.set_backend("cpu")

# Force GPU
ob.set_backend("cuda")

# Or use environment variable
# export OPENBOOST_BACKEND=cuda
```

## GPU Performance

GPU acceleration provides significant speedups for larger datasets:

| Dataset Size | Typical Speedup |
|--------------|-----------------|
| <5K samples | ~1x (CPU overhead dominates) |
| 5K-10K | 2-7x |
| 25K+ | 2-3x |
| 100K+ | 5-10x |

!!! tip "Best practices for GPU"
    - Ensure data is `float32` (not `float64`)
    - Use larger datasets (GPU overhead not worth it for <5K samples)
    - GPU shows best speedup at 10K+ samples

## Multi-GPU Training

```python
import openboost as ob

# Use multiple GPUs with Ray (requires ray[default])
model = ob.GradientBoosting(n_trees=100, n_gpus=4)
model.fit(X, y)

# Or specify exact GPU devices
model = ob.GradientBoosting(n_trees=100, devices=[0, 2])
model.fit(X, y)
```

## Requirements

- NVIDIA GPU with CUDA Compute Capability 3.5+
- CUDA Toolkit 11.0+ or 12.0+
- Numba 0.60+ (`pip install numba`)

## Troubleshooting

### Training seems slow on GPU

- Ensure data is `float32` (not `float64`)
- Use larger datasets (GPU overhead not worth it for <5K samples)
- GPU shows best speedup at 10K+ samples

### Model trained on GPU, loading on CPU machine

```python
# Models are saved in a backend-agnostic format
model.save("model.joblib")

# Load on any machine (CPU or GPU)
loaded = ob.GradientBoosting.load("model.joblib")
```

### CUDA not detected

1. Check CUDA installation: `nvcc --version`
2. Check Numba can see GPU: `python -c "from numba import cuda; print(cuda.gpus)"`
3. Ensure compatible CUDA version with Numba
