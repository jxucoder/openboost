"""OpenBoost benchmarks on Modal (GPU cloud).

Run from Mac with:
    cd openboost
    uv run modal run benchmarks/modal_bench.py

This will execute benchmarks on a cloud A100 GPU.
"""

import modal
from pathlib import Path

# Get the project root (parent of benchmarks/)
PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("openboost-bench")

# Image with CUDA + OpenBoost + Py-Boost dependencies + local source code
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "cupy-cuda12x>=13.0",
        "scikit-learn>=1.0",
        "xgboost>=2.0",
        "joblib>=1.3",
        # Py-Boost (install from PyPI)
        "py-boost",
    )
    # Mount local openboost source code
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost",
    )
)


@app.function(gpu="A100", image=image, timeout=600)
def benchmark_histogram(n_samples: int = 1_000_000, n_features: int = 100):
    """Benchmark histogram building on A100.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        
    Returns:
        Benchmark results dict
    """
    import sys
    sys.path.insert(0, "/root")
    
    from numba import cuda
    import numpy as np
    import time
    
    print(f"Benchmarking histogram: {n_samples:,} samples × {n_features} features")
    print(f"GPU: {cuda.get_current_device().name}")
    
    # Generate data
    np.random.seed(42)
    binned = np.random.randint(0, 256, (n_features, n_samples), dtype=np.uint8)
    grad = np.random.randn(n_samples).astype(np.float32)
    hess = np.abs(np.random.randn(n_samples)).astype(np.float32) + 0.1
    
    # Transfer to GPU
    binned_gpu = cuda.to_device(binned)
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    
    # Allocate output
    hist_grad = cuda.device_array((n_features, 256), dtype=np.float64)
    hist_hess = cuda.device_array((n_features, 256), dtype=np.float64)
    
    # Warmup
    from openboost._backends._cuda import _histogram_kernel
    _histogram_kernel[n_features, 256](binned_gpu, grad_gpu, hess_gpu, hist_grad, hist_hess)
    cuda.synchronize()
    
    # Benchmark
    n_iterations = 100
    cuda.synchronize()
    start = time.perf_counter()
    
    for _ in range(n_iterations):
        _histogram_kernel[n_features, 256](binned_gpu, grad_gpu, hess_gpu, hist_grad, hist_hess)
    
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    time_per_iter_ms = (elapsed / n_iterations) * 1000
    throughput_gbs = (binned.nbytes + grad.nbytes + hess.nbytes) / elapsed / n_iterations / 1e9
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_iterations": n_iterations,
        "total_time_s": elapsed,
        "time_per_iter_ms": time_per_iter_ms,
        "throughput_gb_s": throughput_gbs,
    }


@app.function(gpu="A100", image=image, timeout=1800)
def benchmark_fit_tree(n_samples: int = 100_000, n_features: int = 50, n_rounds: int = 100):
    """Benchmark tree fitting on A100.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_rounds: Number of boosting rounds
        
    Returns:
        Benchmark results dict
    """
    import sys
    sys.path.insert(0, "/root")
    
    from numba import cuda
    import numpy as np
    import time
    import openboost as ob
    
    print(f"Benchmarking fit_tree: {n_samples:,} samples × {n_features} features × {n_rounds} rounds")
    print(f"GPU: {cuda.get_current_device().name}")
    print(f"Backend: {ob.get_backend()}")
    
    # Generate regression data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples) * 0.1
    y = y.astype(np.float32)
    
    # Bin data (one-time cost)
    start_bin = time.perf_counter()
    X_binned = ob.array(X)
    cuda.synchronize()
    bin_time = time.perf_counter() - start_bin
    
    # Train
    pred = np.zeros(n_samples, dtype=np.float32)
    
    cuda.synchronize()
    start_train = time.perf_counter()
    
    for i in range(n_rounds):
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32) * 2
        
        tree = ob.fit_tree(X_binned, grad, hess, max_depth=6)
        tree_pred = tree(X_binned)
        
        # Handle GPU vs CPU prediction
        if hasattr(tree_pred, 'copy_to_host'):
            tree_pred = tree_pred.copy_to_host()
        
        pred = pred + 0.1 * tree_pred
    
    cuda.synchronize()
    train_time = time.perf_counter() - start_train
    
    final_loss = float(np.mean((pred - y) ** 2))
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_rounds": n_rounds,
        "bin_time_s": bin_time,
        "train_time_s": train_time,
        "time_per_round_ms": (train_time / n_rounds) * 1000,
        "final_mse": final_loss,
        "trees_per_second": n_rounds / train_time,
    }


@app.function(gpu="A100", image=image, timeout=3600)
def benchmark_vs_xgboost(n_samples: int = 1_000_000, n_features: int = 100, n_rounds: int = 100):
    """Compare OpenBoost vs XGBoost on A100 with large dataset."""
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import time
    import xgboost as xgb
    from numba import cuda
    import openboost as ob
    
    print(f"\n{'='*60}")
    print(f"OpenBoost vs XGBoost: {n_samples:,} samples × {n_features} features × {n_rounds} rounds")
    print(f"GPU: {cuda.get_current_device().name}")
    print(f"{'='*60}")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples).astype(np.float32) * 0.1
    
    results = {}
    
    # ========== XGBoost GPU ==========
    print("\n[XGBoost GPU]")
    dtrain = xgb.DMatrix(X, label=y)
    params = {
        "max_depth": 6,
        "eta": 0.1,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
    }
    
    # Warmup
    xgb.train(params, dtrain, num_boost_round=5)
    cuda.synchronize()
    
    start = time.perf_counter()
    model = xgb.train(params, dtrain, num_boost_round=n_rounds)
    cuda.synchronize()
    xgb_time = time.perf_counter() - start
    
    pred_xgb = model.predict(dtrain)
    xgb_mse = float(np.mean((pred_xgb - y) ** 2))
    
    print(f"  Time: {xgb_time:.2f}s ({xgb_time/n_rounds*1000:.1f} ms/round)")
    print(f"  MSE: {xgb_mse:.6f}")
    
    results["xgboost"] = {"time_s": xgb_time, "mse": xgb_mse, "ms_per_round": xgb_time/n_rounds*1000}
    
    # ========== OpenBoost GPU ==========
    print("\n[OpenBoost GPU]")
    
    # Warmup
    X_binned = ob.array(X[:10000])
    pred_warmup = np.zeros(10000, dtype=np.float32)
    grad = (2 * pred_warmup).astype(np.float32)
    hess = np.ones(10000, dtype=np.float32) * 2
    tree = ob.fit_tree(X_binned, grad, hess, max_depth=6)
    cuda.synchronize()
    
    # Actual benchmark
    start_bin = time.perf_counter()
    X_binned = ob.array(X)
    cuda.synchronize()
    bin_time = time.perf_counter() - start_bin
    
    pred = np.zeros(n_samples, dtype=np.float32)
    cuda.synchronize()
    start_train = time.perf_counter()
    
    for i in range(n_rounds):
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32) * 2
        tree = ob.fit_tree(X_binned, grad, hess, max_depth=6)
        tree_pred = tree(X_binned)
        if hasattr(tree_pred, 'copy_to_host'):
            tree_pred = tree_pred.copy_to_host()
        pred = pred + 0.1 * tree_pred
    
    cuda.synchronize()
    ob_time = time.perf_counter() - start_train
    ob_mse = float(np.mean((pred - y) ** 2))
    
    print(f"  Binning: {bin_time:.2f}s")
    print(f"  Training: {ob_time:.2f}s ({ob_time/n_rounds*1000:.1f} ms/round)")
    print(f"  MSE: {ob_mse:.6f}")
    
    results["openboost"] = {"time_s": ob_time, "bin_time_s": bin_time, "mse": ob_mse, "ms_per_round": ob_time/n_rounds*1000}
    
    # ========== Summary ==========
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    ratio = ob_time / xgb_time
    print(f"XGBoost:   {xgb_time:.2f}s total, {xgb_time/n_rounds*1000:.1f} ms/round")
    print(f"OpenBoost: {ob_time:.2f}s total, {ob_time/n_rounds*1000:.1f} ms/round")
    print(f"Ratio:     OpenBoost is {ratio:.1f}x {'slower' if ratio > 1 else 'faster'} than XGBoost")
    print(f"\nNote: OpenBoost Phase 2 (P1: GPU kernels for sync elimination)")
    
    results["ratio"] = ratio
    return results


@app.function(gpu="A100", image=image, timeout=3600)
def benchmark_all_frameworks(n_samples: int = 100_000, n_features: int = 50, n_rounds: int = 100):
    """Compare OpenBoost vs Py-Boost vs XGBoost on A100.
    
    This is the main benchmark to compare all three frameworks:
    - XGBoost (C++, highly optimized)
    - Py-Boost (Python/CuPy, GPU-native)
    - OpenBoost (Python/Numba, GPU-native)
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import time
    from numba import cuda
    import warnings
    warnings.filterwarnings('ignore')
    
    print(f"\n{'='*70}")
    print(f"OpenBoost vs Py-Boost vs XGBoost Benchmark")
    print(f"{'='*70}")
    print(f"Config: {n_samples:,} samples × {n_features} features × {n_rounds} rounds")
    print(f"GPU: {cuda.get_current_device().name}")
    print(f"{'='*70}")
    
    # Generate regression data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] ** 2 - 0.3 * X[:, 2] + 
         0.1 * np.random.randn(n_samples)).astype(np.float32)
    
    results = {"config": {"n_samples": n_samples, "n_features": n_features, "n_rounds": n_rounds}}
    
    # ========== 1. XGBoost GPU ==========
    print("\n" + "=" * 50)
    print("[1] XGBoost GPU (tree_method='hist', device='cuda')")
    print("=" * 50)
    
    import xgboost as xgb
    
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {
        "max_depth": 6,
        "eta": 0.1,
        "lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
    }
    
    # Warmup
    xgb.train(xgb_params, dtrain, num_boost_round=5)
    cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=n_rounds)
    cuda.synchronize()
    xgb_time = time.perf_counter() - start
    
    pred_xgb = xgb_model.predict(dtrain)
    xgb_mse = float(np.mean((pred_xgb - y) ** 2))
    
    print(f"  Total time:     {xgb_time:.3f}s")
    print(f"  Time per round: {xgb_time/n_rounds*1000:.2f} ms")
    print(f"  Trees/second:   {n_rounds/xgb_time:.1f}")
    print(f"  Final MSE:      {xgb_mse:.6f}")
    
    results["xgboost"] = {
        "time_s": xgb_time,
        "ms_per_round": xgb_time / n_rounds * 1000,
        "trees_per_sec": n_rounds / xgb_time,
        "mse": xgb_mse,
    }
    
    # ========== 2. Py-Boost GPU ==========
    print("\n" + "=" * 50)
    print("[2] Py-Boost GPU (CuPy-based)")
    print("=" * 50)
    
    try:
        from py_boost.gpu.boosting import GradientBoosting as PyBoostGB
        
        # Warmup
        pb_model_warmup = PyBoostGB(
            loss='mse',
            ntrees=5,
            lr=0.1,
            max_depth=6,
            lambda_l2=1.0,
            verbose=0,
            es=0,  # No early stopping
        )
        pb_model_warmup.fit(X[:10000], y[:10000].reshape(-1, 1))
        cuda.synchronize()
        
        # Benchmark
        pb_model = PyBoostGB(
            loss='mse',
            ntrees=n_rounds,
            lr=0.1,
            max_depth=6,
            lambda_l2=1.0,
            verbose=0,
            es=0,  # No early stopping
        )
        
        start = time.perf_counter()
        pb_model.fit(X, y.reshape(-1, 1))
        cuda.synchronize()
        pb_time = time.perf_counter() - start
        
        pred_pb = pb_model.predict(X).flatten()
        pb_mse = float(np.mean((pred_pb - y) ** 2))
        
        print(f"  Total time:     {pb_time:.3f}s")
        print(f"  Time per round: {pb_time/n_rounds*1000:.2f} ms")
        print(f"  Trees/second:   {n_rounds/pb_time:.1f}")
        print(f"  Final MSE:      {pb_mse:.6f}")
        
        results["pyboost"] = {
            "time_s": pb_time,
            "ms_per_round": pb_time / n_rounds * 1000,
            "trees_per_sec": n_rounds / pb_time,
            "mse": pb_mse,
        }
    except Exception as e:
        print(f"  ERROR: {e}")
        results["pyboost"] = {"error": str(e)}
        pb_time = float('inf')
    
    # ========== 3. OpenBoost GPU ==========
    print("\n" + "=" * 50)
    print("[3] OpenBoost GPU (Numba-based)")
    print("=" * 50)
    
    import openboost as ob
    
    # Warmup
    X_binned_warmup = ob.array(X[:10000])
    grad_warmup = np.zeros(10000, dtype=np.float32)
    hess_warmup = np.ones(10000, dtype=np.float32) * 2
    ob.fit_tree(X_binned_warmup, grad_warmup, hess_warmup, max_depth=6)
    cuda.synchronize()
    
    # Bin data (measure separately)
    start_bin = time.perf_counter()
    X_binned = ob.array(X)
    cuda.synchronize()
    bin_time = time.perf_counter() - start_bin
    
    # Training loop
    pred = np.zeros(n_samples, dtype=np.float32)
    cuda.synchronize()
    start = time.perf_counter()
    
    for i in range(n_rounds):
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(n_samples, dtype=np.float32) * 2
        tree = ob.fit_tree(X_binned, grad, hess, max_depth=6, reg_lambda=1.0)
        tree_pred = tree(X_binned)
        if hasattr(tree_pred, 'copy_to_host'):
            tree_pred = tree_pred.copy_to_host()
        pred = pred + 0.1 * tree_pred
    
    cuda.synchronize()
    ob_time = time.perf_counter() - start
    ob_mse = float(np.mean((pred - y) ** 2))
    
    print(f"  Binning time:   {bin_time:.3f}s")
    print(f"  Training time:  {ob_time:.3f}s")
    print(f"  Time per round: {ob_time/n_rounds*1000:.2f} ms")
    print(f"  Trees/second:   {n_rounds/ob_time:.1f}")
    print(f"  Final MSE:      {ob_mse:.6f}")
    
    results["openboost"] = {
        "bin_time_s": bin_time,
        "time_s": ob_time,
        "ms_per_round": ob_time / n_rounds * 1000,
        "trees_per_sec": n_rounds / ob_time,
        "mse": ob_mse,
    }
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY: Training Time Comparison (excluding binning)")
    print("=" * 70)
    
    # Calculate ratios relative to XGBoost
    xgb_ratio = 1.0
    pb_ratio = pb_time / xgb_time if pb_time != float('inf') else float('inf')
    ob_ratio = ob_time / xgb_time
    
    print(f"{'Framework':<15} {'Time (s)':<12} {'ms/round':<12} {'trees/s':<12} {'vs XGBoost':<12}")
    print("-" * 63)
    print(f"{'XGBoost':<15} {xgb_time:<12.3f} {xgb_time/n_rounds*1000:<12.2f} {n_rounds/xgb_time:<12.1f} {'1.00x (ref)':<12}")
    
    if pb_time != float('inf'):
        pb_status = f"{pb_ratio:.2f}x {'slower' if pb_ratio > 1 else 'faster'}"
        print(f"{'Py-Boost':<15} {pb_time:<12.3f} {pb_time/n_rounds*1000:<12.2f} {n_rounds/pb_time:<12.1f} {pb_status:<12}")
    else:
        print(f"{'Py-Boost':<15} {'ERROR':<12}")
    
    ob_status = f"{ob_ratio:.2f}x {'slower' if ob_ratio > 1 else 'faster'}"
    print(f"{'OpenBoost':<15} {ob_time:<12.3f} {ob_time/n_rounds*1000:<12.2f} {n_rounds/ob_time:<12.1f} {ob_status:<12}")
    
    # Comparison between OpenBoost and Py-Boost
    if pb_time != float('inf'):
        ob_vs_pb = ob_time / pb_time
        print("\n" + "-" * 63)
        print(f"OpenBoost vs Py-Boost: {ob_vs_pb:.2f}x {'slower' if ob_vs_pb > 1 else 'faster'}")
    
    results["ratios"] = {
        "openboost_vs_xgboost": ob_ratio,
        "pyboost_vs_xgboost": pb_ratio if pb_time != float('inf') else None,
        "openboost_vs_pyboost": ob_time / pb_time if pb_time != float('inf') else None,
    }
    
    return results


@app.function(gpu="A100", image=image, timeout=3600)
def benchmark_single_tree_comparison(n_samples: int = 100_000, n_features: int = 50):
    """Compare single tree building time across all frameworks.
    
    This isolates just the tree building step to compare raw performance.
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import time
    from numba import cuda
    import warnings
    warnings.filterwarnings('ignore')
    
    print(f"\n{'='*70}")
    print(f"Single Tree Building Comparison")
    print(f"{'='*70}")
    print(f"Config: {n_samples:,} samples × {n_features} features, max_depth=6")
    print(f"GPU: {cuda.get_current_device().name}")
    print(f"{'='*70}")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    grad = -y.copy()  # For MSE loss starting from pred=0
    hess = np.ones(n_samples, dtype=np.float32)
    
    results = {}
    n_iter = 50  # Build 50 trees and average
    
    # ========== 1. XGBoost ==========
    print("\n[1] XGBoost (single tree)")
    import xgboost as xgb
    
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {
        "max_depth": 6,
        "eta": 1.0,  # Full step
        "lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda",
    }
    
    # Warmup
    xgb.train(xgb_params, dtrain, num_boost_round=1)
    cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iter):
        xgb.train(xgb_params, dtrain, num_boost_round=1)
    cuda.synchronize()
    xgb_time = (time.perf_counter() - start) / n_iter
    
    print(f"  Time per tree: {xgb_time*1000:.3f} ms")
    results["xgboost_ms"] = xgb_time * 1000
    
    # ========== 2. Py-Boost ==========
    print("\n[2] Py-Boost (single tree)")
    
    try:
        from py_boost.gpu.boosting import GradientBoosting as PyBoostGB
        
        # Warmup
        pb_model = PyBoostGB(loss='mse', ntrees=1, lr=1.0, max_depth=6, lambda_l2=1.0, verbose=0, es=0)
        pb_model.fit(X[:10000], y[:10000].reshape(-1, 1))
        cuda.synchronize()
        
        # Benchmark (Py-Boost doesn't allow single-tree timing easily, so we measure ntrees=1)
        start = time.perf_counter()
        for _ in range(n_iter):
            pb_model = PyBoostGB(loss='mse', ntrees=1, lr=1.0, max_depth=6, lambda_l2=1.0, verbose=0, es=0)
            pb_model.fit(X, y.reshape(-1, 1))
        cuda.synchronize()
        pb_time = (time.perf_counter() - start) / n_iter
        
        print(f"  Time per tree: {pb_time*1000:.3f} ms")
        results["pyboost_ms"] = pb_time * 1000
    except Exception as e:
        print(f"  ERROR: {e}")
        results["pyboost_ms"] = None
        pb_time = float('inf')
    
    # ========== 3. OpenBoost ==========
    print("\n[3] OpenBoost (single tree)")
    import openboost as ob
    
    # Warmup
    X_binned = ob.array(X)
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iter):
        ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6, reg_lambda=1.0)
    cuda.synchronize()
    ob_time = (time.perf_counter() - start) / n_iter
    
    print(f"  Time per tree: {ob_time*1000:.3f} ms")
    results["openboost_ms"] = ob_time * 1000
    
    # ========== Summary ==========
    print("\n" + "=" * 70)
    print("SUMMARY: Single Tree Building Time")
    print("=" * 70)
    print(f"{'Framework':<15} {'Time (ms)':<15} {'vs XGBoost':<15}")
    print("-" * 45)
    print(f"{'XGBoost':<15} {xgb_time*1000:<15.3f} {'1.00x (ref)':<15}")
    
    if pb_time != float('inf'):
        pb_ratio = pb_time / xgb_time
        print(f"{'Py-Boost':<15} {pb_time*1000:<15.3f} {pb_ratio:.2f}x {'slower' if pb_ratio > 1 else 'faster':<15}")
    
    ob_ratio = ob_time / xgb_time
    print(f"{'OpenBoost':<15} {ob_time*1000:<15.3f} {ob_ratio:.2f}x {'slower' if ob_ratio > 1 else 'faster':<15}")
    
    if pb_time != float('inf'):
        ob_vs_pb = ob_time / pb_time
        print(f"\nOpenBoost vs Py-Boost: {ob_vs_pb:.2f}x {'slower' if ob_vs_pb > 1 else 'faster'}")
    
    results["ratios"] = {
        "openboost_vs_xgboost": ob_time / xgb_time,
        "pyboost_vs_xgboost": pb_time / xgb_time if pb_time != float('inf') else None,
        "openboost_vs_pyboost": ob_time / pb_time if pb_time != float('inf') else None,
    }
    
    return results


@app.function(gpu="A100", image=image, timeout=3600)
def benchmark_phase2_kernels():
    """Benchmark Phase 2 P1 GPU kernels individually."""
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import time
    from numba import cuda
    
    print(f"\n{'='*60}")
    print("Phase 2 P1 Kernel Benchmarks")
    print(f"GPU: {cuda.get_current_device().name}")
    print(f"{'='*60}")
    
    results = {}
    
    # Setup test data
    n_samples = 100_000
    n_features = 50
    np.random.seed(42)
    
    grad = np.random.randn(n_samples).astype(np.float32)
    hess = np.abs(np.random.randn(n_samples)) .astype(np.float32) + 0.1
    binned = np.random.randint(0, 256, (n_features, n_samples), dtype=np.uint8)
    sample_indices = np.arange(n_samples, dtype=np.int32)
    
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    binned_gpu = cuda.to_device(binned)
    indices_gpu = cuda.to_device(sample_indices)
    
    # Import kernels
    from openboost._backends._cuda import (
        reduce_sum_indexed_cuda,
        gather_cuda,
        partition_samples_cuda,
    )
    
    n_iterations = 100
    
    # ========== Test 1: Reduce Sum ==========
    print("\n[1] reduce_sum_indexed_cuda")
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        result = reduce_sum_indexed_cuda(grad_gpu, indices_gpu)
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"    Time: {elapsed/n_iterations*1000:.3f} ms/call")
    results["reduce_sum_ms"] = elapsed/n_iterations*1000
    
    # Verify correctness
    result_val = result.copy_to_host()[0]
    expected = np.sum(grad)
    print(f"    Result: {result_val:.4f}, Expected: {expected:.4f}, Match: {np.isclose(result_val, expected)}")
    
    # ========== Test 2: Gather 1D ==========
    print("\n[2] gather_cuda (1D)")
    subset_size = 50_000
    subset_indices = cuda.to_device(np.random.choice(n_samples, subset_size, replace=False).astype(np.int32))
    
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        gathered = gather_cuda(grad_gpu, subset_indices)
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"    Time: {elapsed/n_iterations*1000:.3f} ms/call (gathering {subset_size:,} elements)")
    results["gather_1d_ms"] = elapsed/n_iterations*1000
    
    # ========== Test 3: Gather 2D ==========
    print("\n[3] gather_cuda (2D)")
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        gathered = gather_cuda(binned_gpu, subset_indices)
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"    Time: {elapsed/n_iterations*1000:.3f} ms/call ({n_features} features × {subset_size:,} samples)")
    results["gather_2d_ms"] = elapsed/n_iterations*1000
    
    # ========== Test 4: Partition ==========
    print("\n[4] partition_samples_cuda")
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iterations):
        left, right, n_left, n_right = partition_samples_cuda(
            binned_gpu, indices_gpu, feature=0, threshold=128
        )
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"    Time: {elapsed/n_iterations*1000:.3f} ms/call")
    print(f"    Split: {n_left:,} left, {n_right:,} right")
    results["partition_ms"] = elapsed/n_iterations*1000
    
    print(f"\n{'='*60}")
    return results


@app.function(gpu="A100", image=image, timeout=3600)
def benchmark_batch_training(n_configs: int = 100, n_samples: int = 50_000, n_features: int = 20, n_rounds: int = 10):
    """Benchmark batch training of multiple configs (Phase 2 P2).
    
    Compares:
    1. Sequential training (one config at a time)
    2. OpenBoost batch training (all configs in parallel)
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import time
    from numba import cuda
    import openboost as ob
    
    print(f"\n{'='*60}")
    print(f"Batch Training Benchmark: {n_configs} configs × {n_samples:,} samples × {n_features} features × {n_rounds} rounds")
    print(f"GPU: {cuda.get_current_device().name}")
    print(f"{'='*60}")
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + np.random.randn(n_samples).astype(np.float32) * 0.1
    
    # Bin data once (shared across all configs)
    print("\n[1] Binning data...")
    start = time.perf_counter()
    X_binned = ob.array(X)
    cuda.synchronize()
    bin_time = time.perf_counter() - start
    print(f"    Binning time: {bin_time:.3f}s")
    
    # Create config grid
    configs = ob.ConfigBatch.from_grid(
        max_depth=[4, 5, 6, 7, 8],
        reg_lambda=[0.01, 0.1, 1.0, 10.0],
        learning_rate=[0.05, 0.1, 0.2, 0.3, 0.5],
        n_rounds=n_rounds,
    )
    actual_n_configs = min(n_configs, configs.n_configs)
    print(f"    Config grid: {configs.n_configs} total, using {actual_n_configs}")
    
    # Trim configs if needed
    if actual_n_configs < configs.n_configs:
        configs = ob.ConfigBatch.from_lists(
            max_depths=configs.max_depths[:actual_n_configs],
            reg_lambdas=configs.reg_lambdas[:actual_n_configs],
            min_child_weights=configs.min_child_weights[:actual_n_configs],
            learning_rates=configs.learning_rates[:actual_n_configs],
            n_rounds=n_rounds,
        )
    
    results = {"n_configs": actual_n_configs, "n_samples": n_samples, "n_rounds": n_rounds}
    
    # ========== Sequential Training ==========
    print("\n[2] Sequential Training (baseline)...")
    cuda.synchronize()
    start = time.perf_counter()
    
    sequential_trees = []
    for config_idx in range(actual_n_configs):
        config = configs[config_idx]
        pred = np.zeros(n_samples, dtype=np.float32)
        trees_this_config = []
        
        for round_idx in range(n_rounds):
            grad = (2 * (pred - y)).astype(np.float32)
            hess = np.ones(n_samples, dtype=np.float32) * 2
            
            tree = ob.fit_tree(
                X_binned, grad, hess,
                max_depth=config['max_depth'],
                reg_lambda=config['reg_lambda'],
                min_child_weight=config['min_child_weight'],
            )
            trees_this_config.append(tree)
            
            tree_pred = tree(X_binned)
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred = tree_pred.copy_to_host()
            pred = pred + config['learning_rate'] * tree_pred
        
        sequential_trees.append(trees_this_config)
    
    cuda.synchronize()
    sequential_time = time.perf_counter() - start
    
    print(f"    Total time: {sequential_time:.2f}s")
    print(f"    Per config: {sequential_time/actual_n_configs*1000:.1f}ms")
    print(f"    Trees/second: {actual_n_configs * n_rounds / sequential_time:.1f}")
    
    results["sequential_time_s"] = sequential_time
    results["sequential_ms_per_config"] = sequential_time / actual_n_configs * 1000
    
    # ========== Batch Training ==========
    print("\n[3] Batch Training (Phase 2)...")
    cuda.synchronize()
    start = time.perf_counter()
    
    grad_init = (2 * (np.zeros(n_samples, dtype=np.float32) - y)).astype(np.float32)
    hess_init = np.ones(n_samples, dtype=np.float32) * 2
    
    batch_trees = ob.fit_trees_batch(X_binned, grad_init, hess_init, configs)
    
    cuda.synchronize()
    batch_time = time.perf_counter() - start
    
    print(f"    Total time: {batch_time:.2f}s")
    print(f"    Per config: {batch_time/actual_n_configs*1000:.1f}ms")
    print(f"    Trees/second: {actual_n_configs * n_rounds / batch_time:.1f}")
    
    results["batch_time_s"] = batch_time
    results["batch_ms_per_config"] = batch_time / actual_n_configs * 1000
    
    # ========== Summary ==========
    speedup = sequential_time / batch_time
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Sequential: {sequential_time:.2f}s ({sequential_time/actual_n_configs*1000:.1f}ms per config)")
    print(f"Batch:      {batch_time:.2f}s ({batch_time/actual_n_configs*1000:.1f}ms per config)")
    print(f"Speedup:    {speedup:.2f}x")
    
    results["speedup"] = speedup
    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=1200,
)
def benchmark_phase32_gpu_native(
    n_samples: int = 100_000,
    n_features: int = 50,
):
    """Benchmark GPU-native tree building (Phase 4).
    
    This compares:
    1. fit_tree (auto-dispatches to gpu_native on GPU)
    2. fit_tree_gpu_native (explicit)
    3. fit_tree_symmetric_gpu_native (oblivious trees)
    4. XGBoost single tree
    5. Py-Boost single tree
    """
    import sys
    sys.path.insert(0, "/root")
    
    import openboost as ob
    from numba import cuda
    import numpy as np
    import time
    import xgboost as xgb
    
    print(f"Data: {n_samples:,} samples, {n_features} features")
    print()
    
    results = {}
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    # For gradient boosting, we use gradients
    # grad = y - pred (for MSE loss, starting from pred=0)
    grad = -y.astype(np.float32)
    hess = np.ones(n_samples, dtype=np.float32)
    
    # Bin the data once
    X_binned = ob.array(X, n_bins=256)
    
    # Move to GPU
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    
    n_iter = 20
    
    # ========== Warmup all methods ==========
    print("Warming up...")
    ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6)
    ob.fit_tree_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=6)
    ob.fit_tree_symmetric_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    
    # ========== 1. fit_tree (auto-dispatch, now uses gpu_native) ==========
    print("=" * 50)
    print("[1] fit_tree (auto-dispatch to gpu_native)")
    print("=" * 50)
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    fit_tree_time = (time.perf_counter() - start) / n_iter
    print(f"fit_tree:                 {fit_tree_time*1000:.2f} ms/tree")
    results["fit_tree_ms"] = fit_tree_time * 1000
    
    # ========== 2. fit_tree_gpu_native (explicit) ==========
    print()
    print("=" * 50)
    print("[2] fit_tree_gpu_native (explicit)")
    print("=" * 50)
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        ob.fit_tree_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    gpu_native_time = (time.perf_counter() - start) / n_iter
    print(f"fit_tree_gpu_native:      {gpu_native_time*1000:.2f} ms/tree")
    results["gpu_native_ms"] = gpu_native_time * 1000
    
    # ========== 3.5. fit_tree_symmetric_gpu_native (Phase 3.4) ==========
    print()
    print("=" * 50)
    print("[3.5] fit_tree_symmetric_gpu_native (Phase 3.4 - Oblivious)")
    print("=" * 50)
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        ob.fit_tree_symmetric_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    symmetric_time = (time.perf_counter() - start) / n_iter
    print(f"fit_tree_symmetric:       {symmetric_time*1000:.2f} ms/tree")
    results["symmetric_ms"] = symmetric_time * 1000
    
    # ========== 4. XGBoost single tree ==========
    print()
    print("=" * 50)
    print("[4] XGBoost single tree")
    print("=" * 50)
    
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "max_bin": 256,
        "objective": "reg:squarederror",
    }
    
    # Warmup
    xgb.train(xgb_params, dtrain, num_boost_round=1)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_iter):
        xgb.train(xgb_params, dtrain, num_boost_round=1)
    xgb_time = (time.perf_counter() - start) / n_iter
    print(f"XGBoost single tree:      {xgb_time*1000:.2f} ms/tree")
    results["xgb_ms"] = xgb_time * 1000
    
    # ========== 5. Py-Boost single tree ==========
    print()
    print("=" * 50)
    print("[5] Py-Boost single tree")
    print("=" * 50)
    
    try:
        import cupy as cp
        from py_boost.gpu.boosting import GradientBoosting as PyBoostGB
        
        # Py-Boost wants NumPy inputs (it moves to GPU internally)
        y_pb = y.reshape(-1, 1)
        
        # Warmup
        pb_model = PyBoostGB(
            loss='mse',
            ntrees=1,
            max_depth=6,
            max_bin=256,
            lr=1.0,
            verbose=100,  # Can't be 0 (bug in py-boost)
            es=0,
        )
        pb_model.fit(X, y_pb)
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark (train 1 tree at a time)
        start = time.perf_counter()
        for _ in range(n_iter):
            pb_model = PyBoostGB(
                loss='mse',
                ntrees=1,
                max_depth=6,
                max_bin=256,
                lr=1.0,
                verbose=100,  # Can't be 0 (bug in py-boost)
                es=0,
            )
            pb_model.fit(X, y_pb)
        cp.cuda.Stream.null.synchronize()
        pyboost_time = (time.perf_counter() - start) / n_iter
        print(f"Py-Boost single tree:     {pyboost_time*1000:.2f} ms/tree")
        results["pyboost_ms"] = pyboost_time * 1000
    except Exception as e:
        print(f"Py-Boost error: {e}")
        import traceback
        traceback.print_exc()
        pyboost_time = None
        results["pyboost_ms"] = None
    
    # ========== Summary ==========
    print()
    print("=" * 60)
    print("Phase 4 Summary (auto-dispatch enabled)")
    print("=" * 60)
    print(f"fit_tree (auto-dispatch): {fit_tree_time*1000:.2f} ms")
    print(f"fit_tree_gpu_native:      {gpu_native_time*1000:.2f} ms")
    print(f"fit_tree_symmetric:       {symmetric_time*1000:.2f} ms")
    print(f"XGBoost:                  {xgb_time*1000:.2f} ms")
    if pyboost_time:
        print(f"Py-Boost:                 {pyboost_time*1000:.2f} ms")
    print()
    print(f"fit_tree vs XGBoost:      {fit_tree_time/xgb_time:.2f}x" + (" (FASTER!)" if fit_tree_time < xgb_time else ""))
    print(f"GPU-native vs XGBoost:    {gpu_native_time/xgb_time:.2f}x" + (" (FASTER!)" if gpu_native_time < xgb_time else ""))
    if pyboost_time:
        print(f"GPU-native vs Py-Boost:   {pyboost_time/gpu_native_time:.1f}x faster")
    
    results["vs_xgb"] = gpu_native_time / xgb_time
    if pyboost_time:
        results["vs_pyboost"] = gpu_native_time / pyboost_time
    
    return results


@app.local_entrypoint()
def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("OpenBoost Benchmarks on Modal A100")
    print("=" * 70)
    
    # ========== Framework Comparison Benchmarks ==========
    print("\n" + "=" * 70)
    print("FRAMEWORK COMPARISON: OpenBoost vs Py-Boost vs XGBoost")
    print("=" * 70)
    
    # Single tree comparison (raw tree building speed)
    print("\n🌲 Single Tree Building Comparison")
    print("-" * 50)
    single_tree_results = benchmark_single_tree_comparison.remote(
        n_samples=100_000,
        n_features=50,
    )
    
    # Full training comparison (100 rounds)
    print("\n🏁 Full Training Comparison (100 rounds)")
    print("-" * 50)
    full_comparison = benchmark_all_frameworks.remote(
        n_samples=100_000,
        n_features=50,
        n_rounds=100,
    )
    
    # ========== OpenBoost Internal Benchmarks ==========
    print("\n" + "=" * 70)
    print("OPENBOOST INTERNAL BENCHMARKS")
    print("=" * 70)
    
    # Phase 2 P1 kernel benchmarks
    print("\n📊 Phase 2 P1: GPU Kernel Benchmarks")
    print("-" * 50)
    kernel_results = benchmark_phase2_kernels.remote()
    
    # Phase 2 P2 batch training benchmark
    print("\n🚀 Phase 2 P2: Batch Training (Multi-Config)")
    print("-" * 50)
    batch_results = benchmark_batch_training.remote(
        n_configs=50,
        n_samples=30_000,
        n_features=20,
        n_rounds=5,
    )
    
    # Phase 4 GPU-native tree building (with auto-dispatch)
    print("\n🚀 Phase 4: GPU-Native Tree Building")
    print("-" * 50)
    phase32_results = benchmark_phase32_gpu_native.remote(
        n_samples=100_000,
        n_features=50,
    )
    
    print("\n" + "=" * 70)
    print("All Benchmarks Complete!")
    print("=" * 70)


@app.function(
    image=image,
    gpu="A100",
    timeout=1200,
)
def benchmark_fair_comparison(
    n_samples: int = 100_000,
    n_features: int = 50,
):
    """Fair benchmark including ALL overhead (Phase 4.1).
    
    Previous benchmarks unfairly advantaged OpenBoost by:
    - Pre-binning data (not timed)
    - Pre-computing gradients (not timed)
    - Pre-loading data to GPU (not timed)
    
    This benchmark measures the FULL pipeline for fair comparison.
    """
    import sys
    sys.path.insert(0, "/root")
    
    import openboost as ob
    from numba import cuda
    import numpy as np
    import time
    import xgboost as xgb
    
    print("=" * 70)
    print("Phase 4.1: FAIR Benchmark Comparison")
    print("=" * 70)
    print(f"Data: {n_samples:,} samples, {n_features} features")
    print()
    
    results = {}
    
    # Generate data (same for both)
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    n_iter = 10
    
    # ========================================================
    # SCENARIO 1: Single Tree, Full Pipeline
    # ========================================================
    print("=" * 70)
    print("SCENARIO 1: Single Tree - Full Pipeline (Fair)")
    print("=" * 70)
    print("Measures: binning + GPU transfer + gradients + tree building")
    print()
    
    # --- OpenBoost Full Pipeline ---
    print("OpenBoost (full pipeline):")
    
    # Warmup
    X_binned = ob.array(X, n_bins=256)
    grad = -y.copy()
    hess = np.ones_like(y)
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    
    # Measure each component
    ob_binning_times = []
    ob_transfer_times = []
    ob_grad_times = []
    ob_tree_times = []
    
    for _ in range(n_iter):
        # 1. Binning
        cuda.synchronize()
        t0 = time.perf_counter()
        X_binned = ob.array(X, n_bins=256)
        cuda.synchronize()
        ob_binning_times.append(time.perf_counter() - t0)
        
        # 2. Gradient computation (simulating MSE loss)
        t0 = time.perf_counter()
        pred = np.zeros(n_samples, dtype=np.float32)
        grad = 2 * (pred - y)  # MSE gradient
        hess = np.ones(n_samples, dtype=np.float32) * 2
        ob_grad_times.append(time.perf_counter() - t0)
        
        # 3. GPU transfer
        cuda.synchronize()
        t0 = time.perf_counter()
        grad_gpu = cuda.to_device(grad)
        hess_gpu = cuda.to_device(hess)
        cuda.synchronize()
        ob_transfer_times.append(time.perf_counter() - t0)
        
        # 4. Tree building
        cuda.synchronize()
        t0 = time.perf_counter()
        tree = ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6)
        cuda.synchronize()
        ob_tree_times.append(time.perf_counter() - t0)
    
    ob_binning_avg = np.mean(ob_binning_times) * 1000
    ob_transfer_avg = np.mean(ob_transfer_times) * 1000
    ob_grad_avg = np.mean(ob_grad_times) * 1000
    ob_tree_avg = np.mean(ob_tree_times) * 1000
    ob_total = ob_binning_avg + ob_transfer_avg + ob_grad_avg + ob_tree_avg
    
    print(f"  Binning (ob.array):     {ob_binning_avg:>8.2f} ms")
    print(f"  Gradient computation:   {ob_grad_avg:>8.2f} ms")
    print(f"  GPU transfer:           {ob_transfer_avg:>8.2f} ms")
    print(f"  Tree building:          {ob_tree_avg:>8.2f} ms")
    print(f"  ---------------------------------")
    print(f"  TOTAL:                  {ob_total:>8.2f} ms")
    print()
    
    # --- XGBoost Full Pipeline ---
    print("XGBoost (full pipeline):")
    
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "max_bin": 256,
        "objective": "reg:squarederror",
    }
    
    # Warmup
    dtrain = xgb.DMatrix(X, label=y)
    xgb.train(xgb_params, dtrain, num_boost_round=1)
    
    xgb_dmatrix_times = []
    xgb_train_times = []
    
    for _ in range(n_iter):
        # 1. DMatrix creation (includes binning + GPU transfer)
        t0 = time.perf_counter()
        dtrain = xgb.DMatrix(X, label=y)
        xgb_dmatrix_times.append(time.perf_counter() - t0)
        
        # 2. Training (includes gradient computation + tree building)
        t0 = time.perf_counter()
        xgb.train(xgb_params, dtrain, num_boost_round=1)
        xgb_train_times.append(time.perf_counter() - t0)
    
    xgb_dmatrix_avg = np.mean(xgb_dmatrix_times) * 1000
    xgb_train_avg = np.mean(xgb_train_times) * 1000
    xgb_total = xgb_dmatrix_avg + xgb_train_avg
    
    print(f"  DMatrix creation:       {xgb_dmatrix_avg:>8.2f} ms")
    print(f"  Training:               {xgb_train_avg:>8.2f} ms")
    print(f"  ---------------------------------")
    print(f"  TOTAL:                  {xgb_total:>8.2f} ms")
    print()
    
    ratio_s1 = ob_total / xgb_total
    winner_s1 = "OpenBoost" if ratio_s1 < 1 else "XGBoost"
    speedup_s1 = 1/ratio_s1 if ratio_s1 < 1 else ratio_s1
    print(f">>> SCENARIO 1 WINNER: {winner_s1} ({speedup_s1:.2f}x faster)")
    print()
    
    results["s1_ob_total_ms"] = ob_total
    results["s1_xgb_total_ms"] = xgb_total
    results["s1_ratio"] = ratio_s1
    
    # ========================================================
    # SCENARIO 2: 100 Trees, Iterative (OpenBoost Sweet Spot)
    # ========================================================
    print("=" * 70)
    print("SCENARIO 2: 100 Trees - Iterative Training")
    print("=" * 70)
    print("OpenBoost: Bin once, then 100x (grad + fit_tree)")
    print("XGBoost: 100x (xgb.train with num_boost_round=1)")
    print()
    
    n_trees = 100
    
    # --- OpenBoost: Bin once, iterate ---
    print("OpenBoost (bin once, iterate):")
    
    # Pre-bin (count this once)
    cuda.synchronize()
    t0 = time.perf_counter()
    X_binned = ob.array(X, n_bins=256)
    cuda.synchronize()
    ob_binning_once = (time.perf_counter() - t0) * 1000
    
    # Warmup
    pred = np.zeros(n_samples, dtype=np.float32)
    grad = 2 * (pred - y)
    hess = np.ones(n_samples, dtype=np.float32) * 2
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    
    # Time 100 trees
    cuda.synchronize()
    t0 = time.perf_counter()
    pred = np.zeros(n_samples, dtype=np.float32)
    for _ in range(n_trees):
        grad = 2 * (pred - y)
        hess = np.ones(n_samples, dtype=np.float32) * 2
        grad_gpu = cuda.to_device(grad)
        hess_gpu = cuda.to_device(hess)
        tree = ob.fit_tree(X_binned, grad_gpu, hess_gpu, max_depth=6)
        # In real training, would update pred here
    cuda.synchronize()
    ob_iter_time = (time.perf_counter() - t0) * 1000
    ob_total_s2 = ob_binning_once + ob_iter_time
    
    print(f"  Binning (once):         {ob_binning_once:>8.2f} ms")
    print(f"  100 trees:              {ob_iter_time:>8.2f} ms ({ob_iter_time/n_trees:.2f} ms/tree)")
    print(f"  ---------------------------------")
    print(f"  TOTAL:                  {ob_total_s2:>8.2f} ms")
    print()
    
    # --- XGBoost: Create DMatrix once, train 100x ---
    print("XGBoost (100 separate train calls):")
    
    # Create DMatrix once
    t0 = time.perf_counter()
    dtrain = xgb.DMatrix(X, label=y)
    xgb_dmatrix_once = (time.perf_counter() - t0) * 1000
    
    # Warmup
    xgb.train(xgb_params, dtrain, num_boost_round=1)
    
    # Time 100 trees
    t0 = time.perf_counter()
    for _ in range(n_trees):
        xgb.train(xgb_params, dtrain, num_boost_round=1)
    xgb_iter_time = (time.perf_counter() - t0) * 1000
    xgb_total_s2 = xgb_dmatrix_once + xgb_iter_time
    
    print(f"  DMatrix (once):         {xgb_dmatrix_once:>8.2f} ms")
    print(f"  100 train calls:        {xgb_iter_time:>8.2f} ms ({xgb_iter_time/n_trees:.2f} ms/tree)")
    print(f"  ---------------------------------")
    print(f"  TOTAL:                  {xgb_total_s2:>8.2f} ms")
    print()
    
    ratio_s2 = ob_total_s2 / xgb_total_s2
    winner_s2 = "OpenBoost" if ratio_s2 < 1 else "XGBoost"
    speedup_s2 = 1/ratio_s2 if ratio_s2 < 1 else ratio_s2
    print(f">>> SCENARIO 2 WINNER: {winner_s2} ({speedup_s2:.2f}x faster)")
    print()
    
    results["s2_ob_total_ms"] = ob_total_s2
    results["s2_xgb_total_ms"] = xgb_total_s2
    results["s2_ratio"] = ratio_s2
    
    # ========================================================
    # SCENARIO 3: 100 Trees, XGBoost Batched
    # ========================================================
    print("=" * 70)
    print("SCENARIO 3: 100 Trees - XGBoost Batched Mode")
    print("=" * 70)
    print("OpenBoost: Same as Scenario 2 (no batching available)")
    print("XGBoost: Single xgb.train(num_boost_round=100)")
    print()
    
    # --- XGBoost batched ---
    print("XGBoost (batched):")
    
    # Create DMatrix
    t0 = time.perf_counter()
    dtrain = xgb.DMatrix(X, label=y)
    xgb_dmatrix_s3 = (time.perf_counter() - t0) * 1000
    
    # Warmup
    xgb.train(xgb_params, dtrain, num_boost_round=10)
    
    # Time 100 trees in one call
    t0 = time.perf_counter()
    xgb.train(xgb_params, dtrain, num_boost_round=n_trees)
    xgb_batched_time = (time.perf_counter() - t0) * 1000
    xgb_total_s3 = xgb_dmatrix_s3 + xgb_batched_time
    
    print(f"  DMatrix:                {xgb_dmatrix_s3:>8.2f} ms")
    print(f"  train(100 rounds):      {xgb_batched_time:>8.2f} ms ({xgb_batched_time/n_trees:.2f} ms/tree)")
    print(f"  ---------------------------------")
    print(f"  TOTAL:                  {xgb_total_s3:>8.2f} ms")
    print()
    
    print("OpenBoost (same as Scenario 2):")
    print(f"  TOTAL:                  {ob_total_s2:>8.2f} ms")
    print()
    
    ratio_s3 = ob_total_s2 / xgb_total_s3
    winner_s3 = "OpenBoost" if ratio_s3 < 1 else "XGBoost"
    speedup_s3 = 1/ratio_s3 if ratio_s3 < 1 else ratio_s3
    print(f">>> SCENARIO 3 WINNER: {winner_s3} ({speedup_s3:.2f}x faster)")
    print()
    
    results["s3_ob_total_ms"] = ob_total_s2
    results["s3_xgb_total_ms"] = xgb_total_s3
    results["s3_ratio"] = ratio_s3
    
    # ========================================================
    # SCENARIO 4: 100 Trees with Custom Loss (Fair Comparison)
    # ========================================================
    print("=" * 70)
    print("SCENARIO 4: 100 Trees - Custom Loss (Fair Comparison)")
    print("=" * 70)
    print("OpenBoost: Same as Scenario 2")
    print("XGBoost: Custom objective function (Python callback each round)")
    print()
    
    # Custom MSE objective for XGBoost
    def custom_mse_obj(predt, dtrain):
        y = dtrain.get_label()
        grad = 2 * (predt - y)
        hess = np.ones_like(predt) * 2
        return grad, hess
    
    xgb_params_custom = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "max_bin": 256,
        # No objective - using custom
    }
    
    # Warmup
    xgb.train(xgb_params_custom, dtrain, num_boost_round=10, obj=custom_mse_obj)
    
    # Time 100 trees with custom objective
    t0 = time.perf_counter()
    xgb.train(xgb_params_custom, dtrain, num_boost_round=n_trees, obj=custom_mse_obj)
    xgb_custom_time = (time.perf_counter() - t0) * 1000
    xgb_total_s4 = xgb_dmatrix_s3 + xgb_custom_time  # Reuse DMatrix time
    
    print("XGBoost (custom objective):")
    print(f"  DMatrix:                {xgb_dmatrix_s3:>8.2f} ms")
    print(f"  train(100 rounds):      {xgb_custom_time:>8.2f} ms ({xgb_custom_time/n_trees:.2f} ms/tree)")
    print(f"  ---------------------------------")
    print(f"  TOTAL:                  {xgb_total_s4:>8.2f} ms")
    print()
    
    print("OpenBoost:")
    print(f"  TOTAL:                  {ob_total_s2:>8.2f} ms")
    print()
    
    ratio_s4 = ob_total_s2 / xgb_total_s4
    winner_s4 = "OpenBoost" if ratio_s4 < 1 else "XGBoost"
    speedup_s4 = 1/ratio_s4 if ratio_s4 < 1 else ratio_s4
    print(f">>> SCENARIO 4 WINNER: {winner_s4} ({speedup_s4:.2f}x faster)")
    print()
    
    results["s4_ob_total_ms"] = ob_total_s2
    results["s4_xgb_custom_ms"] = xgb_total_s4
    results["s4_ratio"] = ratio_s4
    
    # ========================================================
    # SUMMARY
    # ========================================================
    print("=" * 70)
    print("FAIR BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Scenario':<45} {'OpenBoost':>12} {'XGBoost':>12} {'Winner':>15}")
    print("-" * 85)
    print(f"{'1. Single tree (full pipeline)':<45} {ob_total:>10.1f}ms {xgb_total:>10.1f}ms {winner_s1:>12} {speedup_s1:.1f}x")
    print(f"{'2. 100 trees (iterative)':<45} {ob_total_s2:>10.1f}ms {xgb_total_s2:>10.1f}ms {winner_s2:>12} {speedup_s2:.1f}x")
    print(f"{'3. 100 trees (XGBoost batched, builtin loss)':<45} {ob_total_s2:>10.1f}ms {xgb_total_s3:>10.1f}ms {winner_s3:>12} {speedup_s3:.1f}x")
    print(f"{'4. 100 trees (XGBoost batched, custom loss)':<45} {ob_total_s2:>10.1f}ms {xgb_total_s4:>10.1f}ms {winner_s4:>12} {speedup_s4:.1f}x")
    print()
    print("Note: Scenario 2 & 4 are custom loss comparisons (OpenBoost's sweet spot)")
    print("      Scenario 3 is XGBoost's sweet spot (built-in loss, fully batched)")
    
    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=1200,
)
def benchmark_gradient_boosting(
    n_samples: int = 100_000,
    n_features: int = 50,
):
    """Benchmark Phase 5 GradientBoosting class vs XGBoost.
    
    Tests the new high-level API with batched training.
    """
    import sys
    sys.path.insert(0, "/root")
    
    import openboost as ob
    from numba import cuda
    import numpy as np
    import time
    import xgboost as xgb
    
    print("=" * 70)
    print("Phase 5: GradientBoosting Batch Mode Benchmark")
    print("=" * 70)
    print(f"Data: {n_samples:,} samples, {n_features} features")
    print()
    
    results = {}
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    n_trees = 100
    
    # ========== Warmup ==========
    print("Warming up...")
    
    # OpenBoost GradientBoosting warmup
    model_warmup = ob.GradientBoosting(n_trees=5, max_depth=6, loss='mse')
    model_warmup.fit(X[:1000], y[:1000])
    cuda.synchronize()
    
    # XGBoost warmup
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "max_bin": 256,
        "objective": "reg:squarederror",
    }
    xgb.train(xgb_params, dtrain, num_boost_round=5)
    
    print("Warmup complete.\n")
    
    # ========== 1. OpenBoost GradientBoosting (built-in MSE) ==========
    print("=" * 50)
    print("[1] OpenBoost GradientBoosting (built-in MSE loss)")
    print("=" * 50)
    
    cuda.synchronize()
    start = time.perf_counter()
    model = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=6,
        learning_rate=0.1,
        loss='mse',
    )
    model.fit(X, y)
    cuda.synchronize()
    ob_builtin_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time:             {ob_builtin_time:>8.2f} ms")
    print(f"  Per tree:               {ob_builtin_time/n_trees:>8.2f} ms")
    results["ob_builtin_ms"] = ob_builtin_time
    
    # ========== 2. OpenBoost GradientBoosting (custom MSE) ==========
    print()
    print("=" * 50)
    print("[2] OpenBoost GradientBoosting (custom MSE loss)")
    print("=" * 50)
    
    def custom_mse(pred, y):
        grad = 2.0 * (pred - y)
        hess = np.full_like(pred, 2.0, dtype=np.float32)
        return grad.astype(np.float32), hess
    
    cuda.synchronize()
    start = time.perf_counter()
    model_custom = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=6,
        learning_rate=0.1,
        loss=custom_mse,
    )
    model_custom.fit(X, y)
    cuda.synchronize()
    ob_custom_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time:             {ob_custom_time:>8.2f} ms")
    print(f"  Per tree:               {ob_custom_time/n_trees:>8.2f} ms")
    results["ob_custom_ms"] = ob_custom_time
    
    # ========== 3. XGBoost (built-in MSE, batched) ==========
    print()
    print("=" * 50)
    print("[3] XGBoost (built-in MSE, batched)")
    print("=" * 50)
    
    start = time.perf_counter()
    dtrain = xgb.DMatrix(X, label=y)
    xgb.train(xgb_params, dtrain, num_boost_round=n_trees)
    xgb_builtin_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time:             {xgb_builtin_time:>8.2f} ms")
    print(f"  Per tree:               {xgb_builtin_time/n_trees:>8.2f} ms")
    results["xgb_builtin_ms"] = xgb_builtin_time
    
    # ========== 4. XGBoost (custom MSE, batched) ==========
    print()
    print("=" * 50)
    print("[4] XGBoost (custom MSE, batched)")
    print("=" * 50)
    
    def xgb_custom_mse(predt, dtrain):
        y = dtrain.get_label()
        grad = 2 * (predt - y)
        hess = np.ones_like(predt) * 2
        return grad, hess
    
    xgb_params_custom = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "max_bin": 256,
    }
    
    start = time.perf_counter()
    xgb.train(xgb_params_custom, dtrain, num_boost_round=n_trees, obj=xgb_custom_mse)
    xgb_custom_time = (time.perf_counter() - start) * 1000
    
    print(f"  Total time:             {xgb_custom_time:>8.2f} ms")
    print(f"  Per tree:               {xgb_custom_time/n_trees:>8.2f} ms")
    results["xgb_custom_ms"] = xgb_custom_time
    
    # ========== Summary ==========
    print()
    print("=" * 70)
    print("PHASE 5 BENCHMARK SUMMARY")
    print("=" * 70)
    print()
    print(f"{'Method':<45} {'Time':>12} {'vs XGBoost':>15}")
    print("-" * 75)
    
    # Built-in loss comparison
    ratio1 = ob_builtin_time / xgb_builtin_time
    if ratio1 < 1:
        comparison1 = f"OB {1/ratio1:.1f}x faster"
    else:
        comparison1 = f"XGB {ratio1:.1f}x faster"
    print(f"{'OpenBoost GradientBoosting (builtin)':<45} {ob_builtin_time:>10.1f}ms {comparison1:>15}")
    print(f"{'XGBoost (builtin, batched)':<45} {xgb_builtin_time:>10.1f}ms {'baseline':>15}")
    
    print()
    
    # Custom loss comparison
    ratio2 = ob_custom_time / xgb_custom_time
    if ratio2 < 1:
        comparison2 = f"OB {1/ratio2:.1f}x faster"
    else:
        comparison2 = f"XGB {ratio2:.1f}x faster"
    print(f"{'OpenBoost GradientBoosting (custom)':<45} {ob_custom_time:>10.1f}ms {comparison2:>15}")
    print(f"{'XGBoost (custom, batched)':<45} {xgb_custom_time:>10.1f}ms {'baseline':>15}")
    
    print()
    print("Target: OpenBoost within 1.3x of XGBoost for batched training")
    
    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=1200,
)
def profile_gradient_boosting(
    n_samples: int = 100_000,
    n_features: int = 50,
):
    """Phase 5.2: Detailed profiling of GradientBoosting training.
    
    Includes:
    1. MSE correctness validation vs XGBoost
    2. Per-component timing breakdown
    3. First tree vs subsequent trees (warmup analysis)
    4. Identify highest ROI optimization targets
    """
    import sys
    sys.path.insert(0, "/root")
    
    import openboost as ob
    from numba import cuda
    import numpy as np
    import time
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    
    print("=" * 70)
    print("Phase 5.2: GradientBoosting E2E Profiling")
    print("=" * 70)
    print(f"Data: {n_samples:,} samples, {n_features} features")
    print()
    
    # Generate data with train/test split for MSE validation
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    n_trees = 100
    learning_rate = 0.1
    max_depth = 6
    
    # ================================================================
    # PART 1: MSE CORRECTNESS VALIDATION
    # ================================================================
    print("=" * 70)
    print("PART 1: MSE Correctness Validation")
    print("=" * 70)
    print()
    
    # Train OpenBoost
    print("Training OpenBoost GradientBoosting...")
    ob_model = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        loss='mse',
    )
    ob_model.fit(X_train, y_train)
    ob_pred_train = ob_model.predict(X_train)
    ob_pred_test = ob_model.predict(X_test)
    cuda.synchronize()
    
    # Train XGBoost
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": max_depth,
        "max_bin": 256,
        "learning_rate": learning_rate,
        "objective": "reg:squarederror",
    }
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=n_trees)
    xgb_pred_train = xgb_model.predict(dtrain)
    xgb_pred_test = xgb_model.predict(dtest)
    
    # Compute MSE
    ob_mse_train = np.mean((ob_pred_train - y_train) ** 2)
    ob_mse_test = np.mean((ob_pred_test - y_test) ** 2)
    xgb_mse_train = np.mean((xgb_pred_train - y_train) ** 2)
    xgb_mse_test = np.mean((xgb_pred_test - y_test) ** 2)
    
    # Prediction difference
    pred_diff_test = np.mean(np.abs(ob_pred_test - xgb_pred_test))
    pred_corr = np.corrcoef(ob_pred_test, xgb_pred_test)[0, 1]
    
    print()
    print(f"{'Metric':<30} {'OpenBoost':>15} {'XGBoost':>15} {'Diff':>15}")
    print("-" * 75)
    print(f"{'Train MSE':<30} {ob_mse_train:>15.6f} {xgb_mse_train:>15.6f} {abs(ob_mse_train - xgb_mse_train):>15.6f}")
    print(f"{'Test MSE':<30} {ob_mse_test:>15.6f} {xgb_mse_test:>15.6f} {abs(ob_mse_test - xgb_mse_test):>15.6f}")
    print(f"{'Pred correlation':<30} {'-':>15} {'-':>15} {pred_corr:>15.6f}")
    print(f"{'Mean |pred diff|':<30} {'-':>15} {'-':>15} {pred_diff_test:>15.6f}")
    print()
    
    mse_rel_diff = abs(ob_mse_test - xgb_mse_test) / xgb_mse_test * 100
    if mse_rel_diff < 5:
        print(f"✅ CORRECTNESS CHECK PASSED (MSE diff: {mse_rel_diff:.2f}% < 5%)")
    else:
        print(f"⚠️  CORRECTNESS CHECK WARNING (MSE diff: {mse_rel_diff:.2f}% >= 5%)")
    
    # ================================================================
    # PART 2: PER-COMPONENT TIMING BREAKDOWN
    # ================================================================
    print()
    print("=" * 70)
    print("PART 2: Per-Component Timing Breakdown")
    print("=" * 70)
    print()
    
    # Re-train with instrumentation
    X_binned = ob.array(X_train, n_bins=256)
    y_gpu = cuda.to_device(y_train)
    n_train = len(y_train)
    
    # Pre-allocate
    pred_gpu = cuda.device_array(n_train, dtype=np.float32)
    
    # Fill with zeros
    @cuda.jit
    def fill_zeros(arr, n):
        idx = cuda.grid(1)
        if idx < n:
            arr[idx] = 0.0
    
    threads = 256
    blocks = (n_train + threads - 1) // threads
    fill_zeros[blocks, threads](pred_gpu, n_train)
    cuda.synchronize()
    
    # Timing arrays
    times_gradient = []
    times_tree = []
    times_predict = []
    times_overhead = []
    
    print("Training with instrumentation...")
    
    for i in range(n_trees):
        # Gradient computation
        cuda.synchronize()
        t0 = time.perf_counter()
        grad_gpu, hess_gpu = ob.mse_gradient(pred_gpu, y_gpu)
        cuda.synchronize()
        t1 = time.perf_counter()
        times_gradient.append((t1 - t0) * 1000)
        
        # Tree building
        tree = ob.fit_tree_gpu_native(
            X_binned, grad_gpu, hess_gpu,
            max_depth=max_depth,
            min_child_weight=1.0,
            reg_lambda=1.0,
        )
        cuda.synchronize()
        t2 = time.perf_counter()
        times_tree.append((t2 - t1) * 1000)
        
        # Prediction update
        from openboost._core._predict import predict_tree_add_gpu
        predict_tree_add_gpu(tree, X_binned, pred_gpu, learning_rate)
        cuda.synchronize()
        t3 = time.perf_counter()
        times_predict.append((t3 - t2) * 1000)
        
        # Overhead (list append, etc.)
        t4 = time.perf_counter()
        times_overhead.append((t4 - t3) * 1000)
    
    # Compute stats
    def stats(arr):
        arr = np.array(arr)
        return {
            'mean': np.mean(arr),
            'std': np.std(arr),
            'first': arr[0],
            'rest_mean': np.mean(arr[1:]) if len(arr) > 1 else arr[0],
        }
    
    grad_stats = stats(times_gradient)
    tree_stats = stats(times_tree)
    pred_stats = stats(times_predict)
    overhead_stats = stats(times_overhead)
    
    total_mean = grad_stats['mean'] + tree_stats['mean'] + pred_stats['mean'] + overhead_stats['mean']
    
    print()
    print(f"{'Component':<25} {'Mean (ms)':>12} {'Std':>10} {'First':>12} {'Rest Mean':>12} {'% Total':>10}")
    print("-" * 85)
    print(f"{'Gradient computation':<25} {grad_stats['mean']:>12.3f} {grad_stats['std']:>10.3f} {grad_stats['first']:>12.3f} {grad_stats['rest_mean']:>12.3f} {grad_stats['mean']/total_mean*100:>9.1f}%")
    print(f"{'Tree building':<25} {tree_stats['mean']:>12.3f} {tree_stats['std']:>10.3f} {tree_stats['first']:>12.3f} {tree_stats['rest_mean']:>12.3f} {tree_stats['mean']/total_mean*100:>9.1f}%")
    print(f"{'Prediction update':<25} {pred_stats['mean']:>12.3f} {pred_stats['std']:>10.3f} {pred_stats['first']:>12.3f} {pred_stats['rest_mean']:>12.3f} {pred_stats['mean']/total_mean*100:>9.1f}%")
    print(f"{'Python overhead':<25} {overhead_stats['mean']:>12.3f} {overhead_stats['std']:>10.3f} {overhead_stats['first']:>12.3f} {overhead_stats['rest_mean']:>12.3f} {overhead_stats['mean']/total_mean*100:>9.1f}%")
    print("-" * 85)
    print(f"{'TOTAL':<25} {total_mean:>12.3f}")
    print()
    
    # ================================================================
    # PART 3: FIRST TREE VS REST (WARMUP ANALYSIS)
    # ================================================================
    print("=" * 70)
    print("PART 3: First Tree vs Rest (Warmup Analysis)")
    print("=" * 70)
    print()
    
    first_total = times_gradient[0] + times_tree[0] + times_predict[0]
    rest_total = np.mean(times_gradient[1:]) + np.mean(times_tree[1:]) + np.mean(times_predict[1:])
    
    print(f"First tree total:  {first_total:.3f} ms")
    print(f"Rest trees mean:   {rest_total:.3f} ms")
    print(f"Warmup overhead:   {first_total - rest_total:.3f} ms ({(first_total/rest_total - 1)*100:.1f}% slower)")
    print()
    
    # ================================================================
    # PART 4: PHASE 6 ROI ANALYSIS
    # ================================================================
    print("=" * 70)
    print("PART 4: Phase 6 ROI Analysis")
    print("=" * 70)
    print()
    
    # Compare with XGBoost
    xgb_per_tree = 3.15  # From previous benchmark
    ob_per_tree = total_mean
    gap_ms = ob_per_tree - xgb_per_tree
    
    print(f"OpenBoost per-tree: {ob_per_tree:.2f} ms")
    print(f"XGBoost per-tree:   {xgb_per_tree:.2f} ms")
    print(f"Gap to close:       {gap_ms:.2f} ms ({ob_per_tree/xgb_per_tree:.1f}x)")
    print()
    
    print("Optimization opportunities (sorted by potential ROI):")
    print()
    
    opportunities = [
        ("Tree building", tree_stats['mean'], "Histogram/split kernel tuning"),
        ("Prediction update", pred_stats['mean'], "Fuse with gradient? Pre-allocate?"),
        ("Gradient computation", grad_stats['mean'], "Already fast, minimal gain"),
        ("Python overhead", overhead_stats['mean'], "Minimal, hard to reduce"),
    ]
    
    print(f"{'Component':<25} {'Time (ms)':>12} {'% of Gap':>12} {'Notes':<30}")
    print("-" * 80)
    for name, time_ms, notes in opportunities:
        pct_gap = time_ms / gap_ms * 100 if gap_ms > 0 else 0
        print(f"{name:<25} {time_ms:>12.2f} {pct_gap:>11.1f}% {notes:<30}")
    
    print()
    print("=" * 70)
    print("RECOMMENDATIONS FOR PHASE 6")
    print("=" * 70)
    print()
    print("1. HIGHEST ROI: Tree building optimization")
    print(f"   - Current: {tree_stats['mean']:.2f} ms/tree")
    print("   - Target:  Reduce histogram/split kernel time")
    print("   - Method:  Profile kernels, tune block/thread config")
    print()
    print("2. MEDIUM ROI: Prediction update optimization")
    print(f"   - Current: {pred_stats['mean']:.2f} ms/tree")
    print("   - Target:  Reduce kernel launch overhead")
    print("   - Method:  Pre-allocate tree arrays, reuse buffers")
    print()
    print("3. LOW ROI: Gradient computation")
    print(f"   - Current: {grad_stats['mean']:.2f} ms/tree")
    print("   - Already efficient, minimal room for improvement")
    
    return {
        "mse_ob_test": ob_mse_test,
        "mse_xgb_test": xgb_mse_test,
        "mse_rel_diff_pct": mse_rel_diff,
        "gradient_mean_ms": grad_stats['mean'],
        "tree_mean_ms": tree_stats['mean'],
        "predict_mean_ms": pred_stats['mean'],
        "overhead_mean_ms": overhead_stats['mean'],
        "total_mean_ms": total_mean,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=1200,
)
def profile_gpu_native_kernels(
    n_samples: int = 100_000,
    n_features: int = 50,
):
    """Profile individual kernels in fit_tree_gpu_native.
    
    Breaks down time spent in:
    1. Histogram building
    2. Split finding
    3. Children creation
    4. Sample partitioning
    5. Leaf value computation
    """
    import sys
    sys.path.insert(0, "/root")
    
    import openboost as ob
    from openboost._backends._cuda import (
        _build_level_histograms_kernel,
        _find_level_splits_kernel,
        _create_children_kernel,
        _partition_samples_kernel,
        _compute_leaf_values_kernel,
    )
    from numba import cuda
    import numpy as np
    import time
    import math
    
    print("=" * 60)
    print("GPU Kernel Profiling - fit_tree_gpu_native")
    print("=" * 60)
    print(f"Data: {n_samples:,} samples, {n_features} features, depth=6")
    print()
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    grad = -y.astype(np.float32)
    hess = np.ones(n_samples, dtype=np.float32)
    
    # Bin the data
    X_binned = ob.array(X, n_bins=256)
    binned_gpu = X_binned.data  # Already on GPU
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    
    # Allocate arrays like build_tree_gpu_native does
    max_depth = 6
    max_nodes = 2**(max_depth + 1) - 1
    threads = 256
    sample_blocks = math.ceil(n_samples / threads)
    
    sample_node_ids = cuda.device_array(n_samples, dtype=np.int32)
    histograms = cuda.device_array((max_nodes, n_features, 256, 2), dtype=np.float32)
    node_features = cuda.device_array(max_nodes, dtype=np.int32)
    node_thresholds = cuda.device_array(max_nodes, dtype=np.int32)
    node_values = cuda.device_array(max_nodes, dtype=np.float32)
    node_left = cuda.device_array(max_nodes, dtype=np.int32)
    node_right = cuda.device_array(max_nodes, dtype=np.int32)
    node_gains = cuda.device_array(max_nodes, dtype=np.float32)
    node_sum_grad = cuda.device_array(max_nodes, dtype=np.float32)
    node_sum_hess = cuda.device_array(max_nodes, dtype=np.float32)
    
    reg_lambda_f32 = np.float32(1.0)
    min_child_weight_f32 = np.float32(1.0)
    min_gain_f32 = np.float32(0.0)
    
    # Initialize
    @cuda.jit
    def _init_tree_arrays(sample_node_ids, node_features, node_values, 
                          node_left, node_right, n_samples, max_nodes):
        idx = cuda.grid(1)
        if idx < n_samples:
            sample_node_ids[idx] = 0
        if idx < max_nodes:
            node_features[idx] = -1
            node_values[idx] = 0.0
            node_left[idx] = -1
            node_right[idx] = -1
    
    init_blocks = max(sample_blocks, math.ceil(max_nodes / threads))
    
    n_iter = 50
    timings = {
        "init": [],
        "histogram": [],
        "split_finding": [],
        "children_creation": [],
        "partition": [],
        "leaf_values": [],
    }
    
    print("Running profiling (50 iterations)...")
    
    for _ in range(n_iter):
        # Initialize
        cuda.synchronize()
        t0 = time.perf_counter()
        _init_tree_arrays[init_blocks, threads](
            sample_node_ids, node_features, node_values,
            node_left, node_right, n_samples, max_nodes
        )
        cuda.synchronize()
        timings["init"].append(time.perf_counter() - t0)
        
        hist_time = 0
        split_time = 0
        children_time = 0
        partition_time = 0
        
        # Build tree level by level
        for depth in range(max_depth):
            level_start = 2**depth - 1
            level_end = 2**(depth + 1) - 1
            n_nodes_at_level = level_end - level_start
            
            # Histogram building
            hist_grid = (n_features, n_nodes_at_level)
            cuda.synchronize()
            t0 = time.perf_counter()
            _build_level_histograms_kernel[hist_grid, 256](
                binned_gpu, grad_gpu, hess_gpu, sample_node_ids,
                level_start, level_end, histograms
            )
            cuda.synchronize()
            hist_time += time.perf_counter() - t0
            
            # Split finding
            cuda.synchronize()
            t0 = time.perf_counter()
            _find_level_splits_kernel[n_nodes_at_level, 256](
                histograms, level_start, level_end,
                reg_lambda_f32, min_child_weight_f32, min_gain_f32,
                node_features, node_thresholds, node_gains,
                node_sum_grad, node_sum_hess
            )
            cuda.synchronize()
            split_time += time.perf_counter() - t0
            
            # Children creation
            children_blocks = math.ceil(n_nodes_at_level / threads)
            cuda.synchronize()
            t0 = time.perf_counter()
            _create_children_kernel[children_blocks, threads](
                level_start, level_end, node_features,
                node_left, node_right
            )
            cuda.synchronize()
            children_time += time.perf_counter() - t0
            
            # Partition
            cuda.synchronize()
            t0 = time.perf_counter()
            _partition_samples_kernel[sample_blocks, threads](
                binned_gpu, sample_node_ids, node_features, node_thresholds,
                node_left, node_right, level_start, level_end
            )
            cuda.synchronize()
            partition_time += time.perf_counter() - t0
        
        timings["histogram"].append(hist_time)
        timings["split_finding"].append(split_time)
        timings["children_creation"].append(children_time)
        timings["partition"].append(partition_time)
        
        # Leaf values
        blocks = math.ceil(max_nodes / threads)
        cuda.synchronize()
        t0 = time.perf_counter()
        _compute_leaf_values_kernel[blocks, threads](
            node_features, node_sum_grad, node_sum_hess,
            reg_lambda_f32, node_values, max_nodes
        )
        cuda.synchronize()
        timings["leaf_values"].append(time.perf_counter() - t0)
    
    # Compute statistics
    print()
    print("=" * 60)
    print("Kernel Timing Breakdown (averaged over 50 iterations)")
    print("=" * 60)
    
    total_time = 0
    results = {}
    
    for name, times in timings.items():
        avg_ms = np.mean(times) * 1000
        std_ms = np.std(times) * 1000
        total_time += avg_ms
        results[name] = avg_ms
        print(f"{name:20s}: {avg_ms:8.3f} ms ± {std_ms:.3f} ms")
    
    print("-" * 60)
    print(f"{'TOTAL':20s}: {total_time:8.3f} ms")
    print()
    
    # Percentage breakdown
    print("=" * 60)
    print("Percentage Breakdown")
    print("=" * 60)
    
    for name, avg_ms in results.items():
        pct = (avg_ms / total_time) * 100
        bar = "█" * int(pct / 2)
        print(f"{name:20s}: {pct:5.1f}% {bar}")
    
    print()
    print("=" * 60)
    print("Optimization Opportunities")
    print("=" * 60)
    
    # Identify top bottlenecks
    sorted_times = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop bottlenecks:")
    for i, (name, avg_ms) in enumerate(sorted_times[:3], 1):
        pct = (avg_ms / total_time) * 100
        print(f"  {i}. {name}: {avg_ms:.2f} ms ({pct:.1f}%)")
        
        if name == "histogram":
            print("     → Consider: Cuckoo hashing for sparse data")
            print("     → Consider: Feature bundling (pack 4 uint8 → uint32)")
        elif name == "partition":
            print("     → Consider: Radix sort instead of scatter")
            print("     → Consider: Coalesced memory access patterns")
        elif name == "split_finding":
            print("     → Consider: Parallel reduction across features")
            print("     → Consider: Early termination for min_gain")
    
    return results


@app.function(
    image=image,
    gpu="A100",
    timeout=1200,
)
def profile_python_overhead(
    n_samples: int = 100_000,
    n_features: int = 50,
):
    """Profile Python/Numba overhead breakdown.
    
    Measures:
    1. Memory allocation (cuda.device_array)
    2. JIT compilation overhead
    3. Python loop overhead
    4. Kernel launch overhead
    """
    import sys
    sys.path.insert(0, "/root")
    
    import openboost as ob
    from numba import cuda
    import numpy as np
    import time
    import math
    
    print("=" * 60)
    print("Python/Numba Overhead Breakdown")
    print("=" * 60)
    print(f"Data: {n_samples:,} samples, {n_features} features")
    print()
    
    # Generate data
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    grad = -y.astype(np.float32)
    hess = np.ones(n_samples, dtype=np.float32)
    
    X_binned = ob.array(X, n_bins=256)
    binned_gpu = X_binned.data
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    
    max_depth = 6
    max_nodes = 2**(max_depth + 1) - 1
    threads = 256
    
    n_iter = 100
    
    # ========== 1. Memory Allocation Overhead ==========
    print("Testing memory allocation overhead...")
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        # Same allocations as build_tree_gpu_native
        _ = cuda.device_array(n_samples, dtype=np.int32)  # sample_node_ids
        _ = cuda.device_array((max_nodes, n_features, 256, 2), dtype=np.float32)  # histograms
        _ = cuda.device_array(max_nodes, dtype=np.int32)  # node_features
        _ = cuda.device_array(max_nodes, dtype=np.int32)  # node_thresholds
        _ = cuda.device_array(max_nodes, dtype=np.float32)  # node_values
        _ = cuda.device_array(max_nodes, dtype=np.int32)  # node_left
        _ = cuda.device_array(max_nodes, dtype=np.int32)  # node_right
        _ = cuda.device_array(max_nodes, dtype=np.float32)  # node_gains
        _ = cuda.device_array(max_nodes, dtype=np.float32)  # node_sum_grad
        _ = cuda.device_array(max_nodes, dtype=np.float32)  # node_sum_hess
    cuda.synchronize()
    alloc_time = (time.perf_counter() - start) / n_iter * 1000
    
    # ========== 2. JIT Compilation (inline @cuda.jit) ==========
    print("Testing JIT compilation overhead...")
    
    def test_inline_jit():
        """This simulates what build_tree_gpu_native does - defining JIT inside function."""
        @cuda.jit
        def _dummy_kernel(arr, n):
            idx = cuda.grid(1)
            if idx < n:
                arr[idx] = 0
        
        blocks = math.ceil(max_nodes / threads)
        arr = cuda.device_array(max_nodes, dtype=np.int32)
        _dummy_kernel[blocks, threads](arr, max_nodes)
    
    # First call - includes compilation
    cuda.synchronize()
    start = time.perf_counter()
    test_inline_jit()
    cuda.synchronize()
    first_jit_time = (time.perf_counter() - start) * 1000
    
    # Subsequent calls (still recompiles because it's inside function!)
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        test_inline_jit()
    cuda.synchronize()
    inline_jit_time = (time.perf_counter() - start) / n_iter * 1000
    
    # ========== 3. Module-Level Kernel (no recompilation) ==========
    print("Testing module-level kernel (no recompile)...")
    
    from openboost._backends._cuda import _init_sample_nodes_kernel
    
    arr = cuda.device_array(n_samples, dtype=np.int32)
    blocks = math.ceil(n_samples / threads)
    
    # Warmup
    _init_sample_nodes_kernel[blocks, threads](arr, n_samples)
    
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _init_sample_nodes_kernel[blocks, threads](arr, n_samples)
    cuda.synchronize()
    module_kernel_time = (time.perf_counter() - start) / n_iter * 1000
    
    # ========== 4. Kernel Launch Overhead (empty kernel) ==========
    print("Testing kernel launch overhead...")
    
    @cuda.jit
    def _empty_kernel():
        pass
    
    # Warmup
    _empty_kernel[1, 1]()
    
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        _empty_kernel[1, 1]()
        _empty_kernel[1, 1]()
        _empty_kernel[1, 1]()
        _empty_kernel[1, 1]()  # 4 kernels per level
    cuda.synchronize()
    launch_per_level = (time.perf_counter() - start) / n_iter * 1000
    launch_6_levels = launch_per_level * 6
    
    # ========== 5. Python Loop Overhead ==========
    print("Testing Python loop overhead...")
    
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        for depth in range(max_depth):
            level_start = 2**depth - 1
            level_end = 2**(depth + 1) - 1
            n_nodes_at_level = level_end - level_start
            hist_grid = (n_features, n_nodes_at_level)
            children_blocks = math.ceil(n_nodes_at_level / threads)
    cuda.synchronize()
    loop_time = (time.perf_counter() - start) / n_iter * 1000
    
    # ========== 6. Full fit_tree_gpu_native ==========
    print("Testing full fit_tree_gpu_native...")
    
    # Warmup
    ob.fit_tree_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=6)
    
    cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_iter):
        ob.fit_tree_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=6)
    cuda.synchronize()
    full_time = (time.perf_counter() - start) / n_iter * 1000
    
    # ========== Results ==========
    print()
    print("=" * 60)
    print("Python/Numba Overhead Breakdown")
    print("=" * 60)
    
    overhead_components = {
        "Memory Allocation (10 arrays)": alloc_time,
        "JIT inside function (recompile)": inline_jit_time,
        "Kernel Launch (4×6 levels)": launch_6_levels,
        "Python loop (6 iterations)": loop_time,
    }
    
    total_overhead = sum(overhead_components.values())
    
    print(f"\n{'Component':<35} {'Time (ms)':<12} {'%':>6}")
    print("-" * 55)
    for name, t in overhead_components.items():
        pct = t / full_time * 100
        print(f"{name:<35} {t:>8.3f} ms   {pct:>5.1f}%")
    
    print("-" * 55)
    print(f"{'Total Measured Overhead':<35} {total_overhead:>8.3f} ms   {total_overhead/full_time*100:>5.1f}%")
    print(f"{'Full fit_tree_gpu_native':<35} {full_time:>8.3f} ms   100.0%")
    print(f"{'GPU Kernels (computed)':<35} {full_time - total_overhead:>8.3f} ms   {(full_time - total_overhead)/full_time*100:>5.1f}%")
    
    print()
    print("=" * 60)
    print("Key Insights")
    print("=" * 60)
    
    print(f"""
Memory Allocation: {alloc_time:.2f} ms per tree
   - 10 cuda.device_array() calls
   - Fix: Pool/reuse arrays between trees
   
Inline JIT (build_tree has @cuda.jit inside!): {inline_jit_time:.2f} ms
   - FIRST CALL: {first_jit_time:.2f} ms (compilation)
   - EACH CALL: {inline_jit_time:.2f} ms (re-dispatching)
   - Fix: Move @cuda.jit to module level (already done for most)
   
Kernel Launch: ~{launch_6_levels/6:.3f} ms per kernel x {6*4 + 3} = {launch_6_levels + launch_per_level*3/4:.2f} ms
   - 4 kernels per level x 6 levels + 3 init/leaf = 27 launches
   - Fix: Fuse kernels, reduce launches
   
Python Loop: {loop_time:.2f} ms
   - for depth in range(6): ... (pure Python)
   - Fix: None easy, this is minimal
""")
    
    # Highlight the inline JIT issue
    print("=" * 60)
    print("CRITICAL ISSUE: Inline @cuda.jit")
    print("=" * 60)
    print(f"""
The _init_nodes kernel is defined INSIDE build_tree_gpu_native:

    def build_tree_gpu_native(...):
        ...
        @cuda.jit  # <- This causes JIT dispatch overhead EVERY call!
        def _init_nodes(features, thresholds, left, right, max_n):
            ...

Even though Numba caches the compiled kernel, the Python decorator
machinery (@cuda.jit) still adds overhead on every function call.

Current: ~{inline_jit_time:.2f} ms overhead per tree
Fix:     Move to module level -> ~{module_kernel_time:.2f} ms
Savings: ~{inline_jit_time - module_kernel_time:.2f} ms per tree
""")
    
    return {
        "alloc_ms": alloc_time,
        "inline_jit_ms": inline_jit_time,
        "launch_ms": launch_6_levels,
        "loop_ms": loop_time,
        "total_overhead_ms": total_overhead,
        "full_time_ms": full_time,
    }


@app.function(
    image=image,
    gpu="A100",
    timeout=1800,  # 30 min for large dataset
)
def profile_large_scale(
    n_samples: int = 1_000_000,
    n_features: int = 100,
    n_trees: int = 100,
):
    """Phase 5.3: Large-scale profiling to find true bottlenecks.
    
    Profiles with 1M samples, 100 features to:
    1. Get realistic GPU utilization
    2. Identify true bottlenecks at scale
    3. Find highest ROI optimizations for Phase 6
    """
    import sys
    sys.path.insert(0, "/root")
    
    import openboost as ob
    from numba import cuda
    import numpy as np
    import time
    import xgboost as xgb
    
    print("=" * 70)
    print("Phase 5.3: Large-Scale Profiling")
    print("=" * 70)
    print(f"Data: {n_samples:,} samples, {n_features} features, {n_trees} trees")
    print(f"GPU: {cuda.get_current_device().name}")
    print()
    
    # ================================================================
    # PART 1: DATA GENERATION & BINNING
    # ================================================================
    print("=" * 70)
    print("PART 1: Data Generation & Binning")
    print("=" * 70)
    print()
    
    np.random.seed(42)
    print("Generating data...")
    t0 = time.perf_counter()
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    t1 = time.perf_counter()
    print(f"  Data generation: {t1-t0:.2f}s")
    
    print("Binning data...")
    t0 = time.perf_counter()
    X_binned = ob.array(X, n_bins=256)
    cuda.synchronize()
    t1 = time.perf_counter()
    bin_time = t1 - t0
    print(f"  Binning: {bin_time:.2f}s ({bin_time/n_features*1000:.1f} ms/feature)")
    print()
    
    # ================================================================
    # PART 2: XGBOOST BASELINE
    # ================================================================
    print("=" * 70)
    print("PART 2: XGBoost Baseline")
    print("=" * 70)
    print()
    
    print("Training XGBoost...")
    dtrain = xgb.DMatrix(X, label=y)
    xgb_params = {
        "tree_method": "hist",
        "device": "cuda",
        "max_depth": 6,
        "max_bin": 256,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
    }
    
    # Warmup
    xgb.train(xgb_params, dtrain, num_boost_round=5)
    cuda.synchronize()
    
    t0 = time.perf_counter()
    xgb_model = xgb.train(xgb_params, dtrain, num_boost_round=n_trees)
    cuda.synchronize()
    xgb_time = time.perf_counter() - t0
    
    xgb_pred = xgb_model.predict(dtrain)
    xgb_mse = float(np.mean((xgb_pred - y) ** 2))
    
    print(f"  XGBoost time: {xgb_time:.2f}s ({xgb_time/n_trees*1000:.2f} ms/tree)")
    print(f"  XGBoost MSE:  {xgb_mse:.6f}")
    print()
    
    # ================================================================
    # PART 3: OPENBOOST - HIGH-LEVEL TIMING
    # ================================================================
    print("=" * 70)
    print("PART 3: OpenBoost High-Level Timing")
    print("=" * 70)
    print()
    
    # Warmup
    grad_warmup = np.zeros(1000, dtype=np.float32)
    hess_warmup = np.ones(1000, dtype=np.float32)
    X_warmup = ob.array(X[:1000], n_bins=256)
    ob.fit_tree_gpu_native(X_warmup, grad_warmup, hess_warmup, max_depth=6)
    cuda.synchronize()
    
    # Use GradientBoosting for fair comparison
    print("Training OpenBoost GradientBoosting...")
    ob_model = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=6,
        learning_rate=0.1,
        loss='mse',
    )
    
    t0 = time.perf_counter()
    ob_model.fit(X_binned, y)
    cuda.synchronize()
    ob_time = time.perf_counter() - t0
    
    ob_pred = ob_model.predict(X_binned)
    ob_mse = float(np.mean((ob_pred - y) ** 2))
    
    print(f"  OpenBoost time: {ob_time:.2f}s ({ob_time/n_trees*1000:.2f} ms/tree)")
    print(f"  OpenBoost MSE:  {ob_mse:.6f}")
    print()
    
    # ================================================================
    # PART 4: OPENBOOST - PER-COMPONENT TIMING
    # ================================================================
    print("=" * 70)
    print("PART 4: Per-Component Timing (10 trees)")
    print("=" * 70)
    print()
    
    # Fresh run with detailed timing
    y_gpu = cuda.to_device(y)
    pred_gpu = cuda.device_array(n_samples, dtype=np.float32)
    
    # Zero init
    @cuda.jit
    def fill_zeros(arr, n):
        idx = cuda.grid(1)
        if idx < n:
            arr[idx] = 0.0
    
    threads = 256
    blocks = (n_samples + threads - 1) // threads
    fill_zeros[blocks, threads](pred_gpu, n_samples)
    cuda.synchronize()
    
    times_gradient = []
    times_tree = []
    times_predict = []
    
    sample_trees = 10  # Only profile 10 trees to save time
    
    for i in range(sample_trees):
        # Gradient
        cuda.synchronize()
        t0 = time.perf_counter()
        grad_gpu, hess_gpu = ob.mse_gradient(pred_gpu, y_gpu)
        cuda.synchronize()
        t1 = time.perf_counter()
        times_gradient.append((t1 - t0) * 1000)
        
        # Tree building
        tree = ob.fit_tree_gpu_native(
            X_binned, grad_gpu, hess_gpu,
            max_depth=6, min_child_weight=1.0, reg_lambda=1.0,
        )
        cuda.synchronize()
        t2 = time.perf_counter()
        times_tree.append((t2 - t1) * 1000)
        
        # Prediction
        from openboost._core._predict import predict_tree_add_gpu
        predict_tree_add_gpu(tree, X_binned, pred_gpu, 0.1)
        cuda.synchronize()
        t3 = time.perf_counter()
        times_predict.append((t3 - t2) * 1000)
    
    grad_mean = np.mean(times_gradient)
    tree_mean = np.mean(times_tree)
    pred_mean = np.mean(times_predict)
    total_mean = grad_mean + tree_mean + pred_mean
    
    print(f"{'Component':<25} {'Mean (ms)':>12} {'% Total':>10}")
    print("-" * 50)
    print(f"{'Gradient computation':<25} {grad_mean:>12.2f} {grad_mean/total_mean*100:>9.1f}%")
    print(f"{'Tree building':<25} {tree_mean:>12.2f} {tree_mean/total_mean*100:>9.1f}%")
    print(f"{'Prediction update':<25} {pred_mean:>12.2f} {pred_mean/total_mean*100:>9.1f}%")
    print("-" * 50)
    print(f"{'TOTAL':<25} {total_mean:>12.2f}")
    print()
    
    # ================================================================
    # PART 5: KERNEL-LEVEL PROFILING
    # ================================================================
    print("=" * 70)
    print("PART 5: Kernel-Level Profiling (1 tree)")
    print("=" * 70)
    print()
    
    # Import internal kernels
    from openboost._backends._cuda import (
        _init_sample_nodes_kernel,
        _init_tree_nodes_kernel,
        _zero_level_histograms_kernel,
        _build_histogram_shared_kernel,
        _find_level_splits_kernel,
        _create_children_kernel,
        _partition_samples_kernel,
        _zero_float_array_kernel,
        _compute_leaf_sums_kernel,
        _compute_leaf_values_kernel,
    )
    import math
    
    max_depth = 6
    max_nodes = 2**(max_depth + 1) - 1
    
    # Re-compute gradients
    grad_gpu, hess_gpu = ob.mse_gradient(pred_gpu, y_gpu)
    cuda.synchronize()
    
    # Get binned data
    binned_gpu = X_binned.data
    
    # Allocate arrays
    sample_node_ids = cuda.device_array(n_samples, dtype=np.int32)
    histograms = cuda.device_array((max_nodes, n_features, 256, 2), dtype=np.float32)
    node_features = cuda.device_array(max_nodes, dtype=np.int32)
    node_thresholds = cuda.device_array(max_nodes, dtype=np.int32)
    node_values = cuda.device_array(max_nodes, dtype=np.float32)
    node_left = cuda.device_array(max_nodes, dtype=np.int32)
    node_right = cuda.device_array(max_nodes, dtype=np.int32)
    node_gains = cuda.device_array(max_nodes, dtype=np.float32)
    node_sum_grad = cuda.device_array(max_nodes, dtype=np.float32)
    node_sum_hess = cuda.device_array(max_nodes, dtype=np.float32)
    
    node_blocks = math.ceil(max_nodes / threads)
    sample_blocks = math.ceil(n_samples / threads)
    
    # Time each kernel
    kernel_times = {}
    
    # Init kernels
    cuda.synchronize()
    t0 = time.perf_counter()
    _init_tree_nodes_kernel[node_blocks, threads](
        node_features, node_thresholds, node_left, node_right, max_nodes
    )
    cuda.synchronize()
    kernel_times['init_tree_nodes'] = (time.perf_counter() - t0) * 1000
    
    cuda.synchronize()
    t0 = time.perf_counter()
    _init_sample_nodes_kernel[sample_blocks, threads](sample_node_ids, n_samples)
    cuda.synchronize()
    kernel_times['init_sample_nodes'] = (time.perf_counter() - t0) * 1000
    
    # Level-by-level kernels
    hist_total = 0
    split_total = 0
    children_total = 0
    partition_total = 0
    
    reg_lambda_f32 = np.float32(1.0)
    min_child_weight_f32 = np.float32(1.0)
    min_gain_f32 = np.float32(0.0)
    
    CHUNK_SIZE = 4096
    n_chunks = math.ceil(n_samples / CHUNK_SIZE)
    
    for depth in range(max_depth):
        level_start = 2**depth - 1
        level_end = 2**(depth + 1) - 1
        n_nodes_at_level = level_end - level_start
        
        # Histogram - Phase 6.3: shared memory approach (100x fewer global atomics)
        cuda.synchronize()
        t0 = time.perf_counter()
        # Zero histograms for this level
        _zero_level_histograms_kernel[(n_nodes_at_level, n_features), 256](
            histograms, level_start, level_end, n_features
        )
        # Build using shared memory kernel
        hist_grid = (n_features, n_chunks)
        if n_nodes_at_level <= 16:
            # Single pass
            _build_histogram_shared_kernel[hist_grid, 256](
                binned_gpu, grad_gpu, hess_gpu, sample_node_ids,
                level_start, n_nodes_at_level, 0,
                histograms
            )
        else:
            # Two passes for depth 5 (32 nodes)
            _build_histogram_shared_kernel[hist_grid, 256](
                binned_gpu, grad_gpu, hess_gpu, sample_node_ids,
                level_start, 16, 0,
                histograms
            )
            remaining_nodes = n_nodes_at_level - 16
            _build_histogram_shared_kernel[hist_grid, 256](
                binned_gpu, grad_gpu, hess_gpu, sample_node_ids,
                level_start, remaining_nodes, 16,
                histograms
            )
        cuda.synchronize()
        hist_total += (time.perf_counter() - t0) * 1000
        
        # Split finding
        cuda.synchronize()
        t0 = time.perf_counter()
        _find_level_splits_kernel[n_nodes_at_level, 256](
            histograms, level_start, level_end,
            reg_lambda_f32, min_child_weight_f32, min_gain_f32,
            node_features, node_thresholds, node_gains,
            node_sum_grad, node_sum_hess
        )
        cuda.synchronize()
        split_total += (time.perf_counter() - t0) * 1000
        
        # Create children
        cuda.synchronize()
        t0 = time.perf_counter()
        children_blocks = math.ceil(n_nodes_at_level / threads)
        _create_children_kernel[children_blocks, threads](
            level_start, level_end, node_features, node_left, node_right
        )
        cuda.synchronize()
        children_total += (time.perf_counter() - t0) * 1000
        
        # Partition
        cuda.synchronize()
        t0 = time.perf_counter()
        _partition_samples_kernel[sample_blocks, threads](
            binned_gpu, sample_node_ids, node_features, node_thresholds,
            node_left, node_right, level_start, level_end
        )
        cuda.synchronize()
        partition_total += (time.perf_counter() - t0) * 1000
    
    kernel_times['histogram_build'] = hist_total
    kernel_times['split_finding'] = split_total
    kernel_times['create_children'] = children_total
    kernel_times['partition_samples'] = partition_total
    
    # Leaf computation (the fixed version)
    cuda.synchronize()
    t0 = time.perf_counter()
    _zero_float_array_kernel[node_blocks, threads](node_sum_grad, max_nodes)
    _zero_float_array_kernel[node_blocks, threads](node_sum_hess, max_nodes)
    cuda.synchronize()
    kernel_times['zero_sums'] = (time.perf_counter() - t0) * 1000
    
    cuda.synchronize()
    t0 = time.perf_counter()
    _compute_leaf_sums_kernel[sample_blocks, threads](
        grad_gpu, hess_gpu, sample_node_ids, node_sum_grad, node_sum_hess
    )
    cuda.synchronize()
    kernel_times['compute_leaf_sums'] = (time.perf_counter() - t0) * 1000
    
    cuda.synchronize()
    t0 = time.perf_counter()
    _compute_leaf_values_kernel[node_blocks, threads](
        node_features, node_sum_grad, node_sum_hess,
        reg_lambda_f32, node_values, max_nodes
    )
    cuda.synchronize()
    kernel_times['compute_leaf_values'] = (time.perf_counter() - t0) * 1000
    
    # Print kernel breakdown
    total_kernel = sum(kernel_times.values())
    
    print(f"{'Kernel':<30} {'Time (ms)':>12} {'% Total':>10}")
    print("-" * 55)
    
    # Sort by time descending
    sorted_kernels = sorted(kernel_times.items(), key=lambda x: x[1], reverse=True)
    for name, time_ms in sorted_kernels:
        pct = time_ms / total_kernel * 100
        print(f"{name:<30} {time_ms:>12.3f} {pct:>9.1f}%")
    
    print("-" * 55)
    print(f"{'TOTAL':<30} {total_kernel:>12.3f}")
    print()
    
    # ================================================================
    # PART 6: SUMMARY & PHASE 6 RECOMMENDATIONS
    # ================================================================
    print("=" * 70)
    print("PART 6: Summary & Phase 6 Recommendations")
    print("=" * 70)
    print()
    
    ob_per_tree = ob_time / n_trees * 1000
    xgb_per_tree = xgb_time / n_trees * 1000
    gap = ob_per_tree - xgb_per_tree
    ratio = ob_per_tree / xgb_per_tree
    
    print(f"{'Metric':<30} {'Value':>15}")
    print("-" * 50)
    print(f"{'OpenBoost per-tree':<30} {ob_per_tree:>12.2f} ms")
    print(f"{'XGBoost per-tree':<30} {xgb_per_tree:>12.2f} ms")
    print(f"{'Gap':<30} {gap:>12.2f} ms")
    print(f"{'Ratio':<30} {ratio:>12.2f}x slower")
    print(f"{'OpenBoost MSE':<30} {ob_mse:>15.6f}")
    print(f"{'XGBoost MSE':<30} {xgb_mse:>15.6f}")
    print()
    
    # Top ROI based on kernel times
    print("TOP 3 ROI FOR PHASE 6:")
    print("-" * 50)
    for i, (name, time_ms) in enumerate(sorted_kernels[:3], 1):
        pct_of_gap = time_ms / gap * 100 if gap > 0 else 0
        print(f"{i}. {name}: {time_ms:.2f} ms ({pct_of_gap:.0f}% of gap)")
    print()
    
    mse_diff_pct = abs(ob_mse - xgb_mse) / xgb_mse * 100
    if mse_diff_pct < 5:
        print(f"✅ MSE CORRECTNESS PASSED (diff: {mse_diff_pct:.2f}%)")
    else:
        print(f"⚠️  MSE CORRECTNESS WARNING (diff: {mse_diff_pct:.2f}%)")
    
    return {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_trees": n_trees,
        "ob_time_s": ob_time,
        "xgb_time_s": xgb_time,
        "ob_per_tree_ms": ob_per_tree,
        "xgb_per_tree_ms": xgb_per_tree,
        "ratio": ratio,
        "ob_mse": ob_mse,
        "xgb_mse": xgb_mse,
        "kernel_times": kernel_times,
    }


@app.function(gpu="A100", image=image, timeout=900)
def benchmark_10m():
    """Benchmark with 10M samples to confirm performance at scale."""
    import sys
    sys.path.insert(0, "/root")
    
    import time
    import numpy as np
    import xgboost as xgb
    from numba import cuda
    
    import openboost as ob
    from openboost._core._tree import fit_tree_gpu_native
    
    print("=" * 70)
    print("10M Sample Benchmark")
    print("=" * 70)
    
    n_samples = 10_000_000
    n_features = 100
    n_trees = 10
    max_depth = 6
    
    print(f"Data: {n_samples:,} samples, {n_features} features, {n_trees} trees")
    print(f"GPU: {cuda.get_current_device().name}")
    print()
    
    # Generate data
    print("Generating data...")
    np.random.seed(42)
    t0 = time.perf_counter()
    X_np = np.random.randn(n_samples, n_features).astype(np.float32)
    y_np = np.random.randn(n_samples).astype(np.float32)
    print(f"  Data generation: {time.perf_counter() - t0:.2f}s")
    
    # XGBoost
    print("\n" + "=" * 70)
    print("XGBoost")
    print("=" * 70)
    
    t0 = time.perf_counter()
    dtrain = xgb.DMatrix(X_np, label=y_np)
    dmatrix_time = time.perf_counter() - t0
    print(f"  DMatrix creation: {dmatrix_time:.2f}s")
    
    params = {
        'max_depth': max_depth,
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror',
    }
    
    # Warmup
    xgb.train(params, dtrain, num_boost_round=1)
    
    t0 = time.perf_counter()
    xgb_model = xgb.train(params, dtrain, num_boost_round=n_trees)
    xgb_time = time.perf_counter() - t0
    xgb_per_tree = xgb_time / n_trees * 1000
    print(f"  Training: {xgb_time:.2f}s ({xgb_per_tree:.2f} ms/tree)")
    
    # OpenBoost
    print("\n" + "=" * 70)
    print("OpenBoost")
    print("=" * 70)
    
    t0 = time.perf_counter()
    X_binned = ob.array(X_np)
    binning_time = time.perf_counter() - t0
    print(f"  Binning: {binning_time:.2f}s")
    
    grad = (y_np * 2).astype(np.float32)
    hess = np.ones(n_samples, dtype=np.float32)
    
    # Move to GPU
    grad_gpu = cuda.to_device(grad)
    hess_gpu = cuda.to_device(hess)
    
    # Warmup
    _ = fit_tree_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=max_depth)
    cuda.synchronize()
    
    # Benchmark
    cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_trees):
        tree = fit_tree_gpu_native(X_binned, grad_gpu, hess_gpu, max_depth=max_depth)
    cuda.synchronize()
    ob_time = time.perf_counter() - t0
    ob_per_tree = ob_time / n_trees * 1000
    print(f"  Training: {ob_time:.2f}s ({ob_per_tree:.2f} ms/tree)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY (10M samples)")
    print("=" * 70)
    print()
    print(f"{'Method':<20} {'Per-tree (ms)':<15} {'vs XGBoost':<15}")
    print("-" * 50)
    print(f"{'XGBoost':<20} {xgb_per_tree:<15.2f} {'1.00x':<15}")
    print(f"{'OpenBoost':<20} {ob_per_tree:<15.2f} {ob_per_tree/xgb_per_tree:.2f}x")
    
    ratio = ob_per_tree / xgb_per_tree
    if ratio < 1:
        print(f"\n✅ OpenBoost is {1/ratio:.2f}x FASTER than XGBoost!")
    else:
        print(f"\n⚠️ OpenBoost is {ratio:.2f}x slower than XGBoost")
    
    return {
        "n_samples": n_samples,
        "xgb_per_tree_ms": xgb_per_tree,
        "ob_per_tree_ms": ob_per_tree,
        "ratio": ratio,
    }


# NOTE: Phase 7 V2 benchmark functions (profile_v1_vs_v2, benchmark_phase7_rowbased)
# were removed after analysis showed V1 is 4x faster than V2.
# See logs/2026-01-03-phase-7-final.md for details.
# End of benchmark file
