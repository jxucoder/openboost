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

GPU benefit depends on dataset shape, tree parameters, distribution, CUDA
stack, transfer policy, and JIT warm-up. OpenBoost does not publish a universal
speedup table without a checked-in benchmark artifact.

For a defensible comparison:

1. force the backend with `ob.set_backend("cpu")` or `"cuda"`;
2. run a warm-up that is excluded from timed repetitions;
3. compare predictions and task metrics before comparing runtime;
4. report repeated fit/predict timings, peak memory, failures, and hardware;
5. save raw results with the OpenBoost commit and dependency versions.

The ScoringBench integration under `benchmarks/scoringbench/` defines separate
official-quality and scale-extension protocols for probabilistic models.

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
- Exclude first-use JIT compilation only when the benchmark protocol says so
- Check that `ob.get_backend()` reports `"cuda"`
- Measure fit and prediction separately; do not assume a crossover dataset size

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
