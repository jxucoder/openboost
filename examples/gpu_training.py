#!/usr/bin/env python
"""GPU training example with OpenBoost.

This example demonstrates:
- Automatic GPU detection
- Manual backend selection
- GPU vs CPU performance comparison
- Best practices for GPU training

Requirements:
- NVIDIA GPU with CUDA support
- Numba with CUDA support: pip install numba
- CUDA Toolkit installed
"""

import time
import numpy as np

# OpenBoost imports
import openboost as ob
from openboost import (
    GradientBoosting,
    NaturalBoostNormal,
    OpenBoostGAM,
)


def generate_large_dataset(n_samples: int = 50000, n_features: int = 20, seed: int = 42):
    """Generate a large dataset for GPU benchmarking."""
    np.random.seed(seed)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Non-linear relationship
    y = (
        2 * X[:, 0]
        + X[:, 1] ** 2
        - 0.5 * X[:, 2] * X[:, 3]
        + np.sin(X[:, 4] * 2)
        + 0.3 * X[:, 5:10].sum(axis=1)
        + np.random.randn(n_samples).astype(np.float32) * 0.5
    )
    return X, y


def time_training(model, X, y, name: str) -> float:
    """Time model training and return duration."""
    start = time.perf_counter()
    model.fit(X, y)
    duration = time.perf_counter() - start
    return duration


def main():
    print("=" * 60)
    print("OpenBoost GPU Training Example")
    print("=" * 60)
    
    # --- Backend Detection ---
    print("\n1. Backend detection...")
    
    current_backend = ob.get_backend()
    is_gpu = ob.is_cuda()
    is_cpu_only = ob.is_cpu()
    
    print(f"   Current backend: {current_backend}")
    print(f"   GPU available: {is_gpu}")
    print(f"   CPU-only mode: {is_cpu_only}")
    
    if is_gpu:
        print("   GPU is available and will be used for training!")
    else:
        print("   Running in CPU mode. For GPU:")
        print("   - Install CUDA toolkit")
        print("   - Install numba with CUDA: pip install numba")
        print("   - Ensure NVIDIA GPU is available")
    
    # --- Generate Data ---
    print("\n2. Generating large dataset...")
    
    n_samples = 50000  # Large enough to see GPU benefit
    n_features = 20
    
    X, y = generate_large_dataset(n_samples, n_features)
    
    # Split
    n_train = int(n_samples * 0.8)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]
    
    print(f"   Samples: {n_samples:,}")
    print(f"   Features: {n_features}")
    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"   Data type: {X.dtype}")
    
    # --- Benchmark GradientBoosting ---
    print("\n3. Benchmarking GradientBoosting...")
    
    model_config = {
        'n_trees': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'loss': 'mse',
    }
    
    # Warmup (first run compiles Numba functions)
    print("   Warmup run (JIT compilation)...")
    warmup_model = GradientBoosting(**model_config)
    warmup_model.fit(X_train[:1000], y_train[:1000])
    
    # Timed run
    print("   Timed run...")
    model = GradientBoosting(**model_config)
    duration = time_training(model, X_train, y_train, "GradientBoosting")
    
    y_pred = model.predict(X_test)
    rmse = ob.rmse_score(y_test, y_pred)
    
    print(f"   Training time: {duration:.2f}s")
    print(f"   Trees/second: {model_config['n_trees'] / duration:.1f}")
    print(f"   Test RMSE: {rmse:.4f}")
    print(f"   Backend used: {ob.get_backend()}")
    
    # --- Benchmark NaturalBoost ---
    print("\n4. Benchmarking NaturalBoostNormal...")
    
    # Warmup
    warmup_prob = NaturalBoostNormal(n_trees=10, max_depth=4)
    warmup_prob.fit(X_train[:1000], y_train[:1000])
    
    # Timed run
    prob_model = NaturalBoostNormal(
        n_trees=50,
        max_depth=5,
        learning_rate=0.1,
    )
    duration_prob = time_training(prob_model, X_train, y_train, "NaturalBoost")
    
    print(f"   Training time: {duration_prob:.2f}s")
    print(f"   Trees/second: {50 * 2 / duration_prob:.1f}")  # 2 param trees per round
    
    # --- Benchmark GAM ---
    print("\n5. Benchmarking OpenBoostGAM...")
    
    # GAM is especially fast on GPU since all features update in parallel
    # Warmup
    warmup_gam = OpenBoostGAM(n_rounds=10, learning_rate=0.1)
    warmup_gam.fit(X_train[:1000], y_train[:1000])
    
    # Timed run
    gam = OpenBoostGAM(
        n_rounds=200,
        learning_rate=0.05,
    )
    duration_gam = time_training(gam, X_train, y_train, "GAM")
    
    y_pred_gam = gam.predict(X_test)
    rmse_gam = ob.rmse_score(y_test, y_pred_gam)
    
    print(f"   Training time: {duration_gam:.2f}s")
    print(f"   Rounds/second: {200 / duration_gam:.1f}")
    print(f"   Test RMSE: {rmse_gam:.4f}")
    
    # --- Manual Backend Selection ---
    print("\n6. Manual backend control...")
    
    print(f"   Current backend: {ob.get_backend()}")
    
    # You can force CPU for debugging
    if is_gpu:
        print("   Switching to CPU mode...")
        ob.set_backend("cpu")
        print(f"   Backend now: {ob.get_backend()}")
        
        # Train on CPU
        cpu_model = GradientBoosting(n_trees=20, max_depth=4)
        duration_cpu = time_training(cpu_model, X_train[:10000], y_train[:10000], "CPU")
        print(f"   CPU training time (10k samples, 20 trees): {duration_cpu:.2f}s")
        
        # Switch back to GPU
        ob.set_backend("cuda")
        print(f"   Backend now: {ob.get_backend()}")
        
        # Train on GPU
        gpu_model = GradientBoosting(n_trees=20, max_depth=4)
        duration_gpu = time_training(gpu_model, X_train[:10000], y_train[:10000], "GPU")
        print(f"   GPU training time (10k samples, 20 trees): {duration_gpu:.2f}s")
        
        if duration_gpu < duration_cpu:
            speedup = duration_cpu / duration_gpu
            print(f"   GPU speedup: {speedup:.1f}x faster")
        else:
            print("   (GPU overhead may dominate for small datasets)")
    else:
        print("   GPU not available, skipping backend comparison")
    
    # --- Best Practices ---
    print("\n7. GPU training best practices...")
    
    best_practices = """
   DATA TYPE:
   - Use float32 (not float64) for all arrays
   - Example: X = X.astype(np.float32)
   - OpenBoost automatically converts, but explicit is better
   
   DATASET SIZE:
   - GPU shows best speedup at 10K+ samples
   - Below 5K samples, CPU may be faster due to overhead
   - GAM benefits most from GPU (parallel feature updates)
   
   MEMORY:
   - GPU memory limits max dataset size
   - For very large data, use mini-batch training
   - Example: model = GradientBoosting(subsample=0.8)
   
   DEBUGGING:
   - Start on CPU: ob.set_backend("cpu")
   - Once working, switch to GPU: ob.set_backend("cuda")
   - Environment variable: OPENBOOST_BACKEND=cuda
   
   MODEL SAVING:
   - Models save in backend-agnostic format
   - Train on GPU, load on CPU machine works!
   - Example:
     model.save("model.joblib")  # On GPU machine
     loaded = GradientBoosting.load("model.joblib")  # Works anywhere
"""
    print(best_practices)
    
    # --- Scaling Guide ---
    print("\n8. Expected GPU speedups by dataset size...")
    
    scaling_info = """
   | Dataset Size | Features | Trees | Expected Speedup |
   |--------------|----------|-------|------------------|
   | 5K samples   | 10       | 100   | ~1-2x            |
   | 10K samples  | 20       | 100   | ~2-5x            |
   | 50K samples  | 20       | 100   | ~3-7x            |
   | 100K samples | 50       | 200   | ~5-10x           |
   | 500K samples | 100      | 500   | ~10-20x          |
   
   Factors affecting speedup:
   - More features = better GPU utilization
   - More bins = better GPU utilization  
   - GAM shows best speedups (parallel feature updates)
   - First run includes JIT compilation overhead
"""
    print(scaling_info)
    
    # --- Multi-GPU (if available) ---
    print("\n9. Multi-GPU training...")
    
    print("""
   For datasets that don't fit on a single GPU or to speed up training further,
   OpenBoost supports multi-GPU training via Ray:
   
   # Install Ray
   pip install ray[default]
   
   # Use multiple GPUs
   model = GradientBoosting(
       n_trees=100,
       max_depth=6,
       n_gpus=4,  # Use 4 GPUs
   )
   model.fit(X, y)
   
   # Or specify exact devices
   model = GradientBoosting(
       n_trees=100,
       devices=[0, 2, 3],  # Use GPUs 0, 2, 3
   )
   
   Multi-GPU training uses data parallelism:
   - Each GPU processes a subset of samples
   - Histograms are aggregated across GPUs
   - Near-linear scaling with number of GPUs
""")
    
    # --- Summary ---
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print(f"""
   Backend: {ob.get_backend()}
   GPU Available: {ob.is_cuda()}
   
   Results on {n_samples:,} samples, {n_features} features:
   - GradientBoosting (100 trees): {duration:.2f}s
   - NaturalBoost (50 trees): {duration_prob:.2f}s  
   - OpenBoostGAM (200 rounds): {duration_gam:.2f}s
""")
    
    print("=" * 60)
    print("Example complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
