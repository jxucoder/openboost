"""Benchmark: OpenBoost GPU-GAM vs InterpretML EBM.

Run locally (if you have GPU):
    cd openboost
    uv run python benchmarks/ebm_benchmark.py

Run on Modal (cloud A100):
    cd openboost
    uv run modal run benchmarks/ebm_benchmark.py
"""

import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("openboost-ebm-bench")

# Image with CUDA + OpenBoost + InterpretML
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "cupy-cuda12x>=13.0",
        "scikit-learn>=1.0",
        "interpret>=0.6",  # InterpretML with EBM
        "xgboost>=2.0",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost",
    )
)


def generate_data(n_samples: int, n_features: int, task: str = "regression"):
    """Generate synthetic data for benchmarking."""
    import numpy as np
    
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    if task == "regression":
        # True model: additive (perfect for GAM)
        y = (
            np.sin(X[:, 0] * 2) +           # Non-linear effect
            0.5 * X[:, 1] +                  # Linear effect
            np.where(X[:, 2] > 0, 0.3, -0.3) +  # Step function
            0.1 * np.random.randn(n_samples)  # Noise
        ).astype(np.float32)
    else:
        # Classification
        logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        probs = 1 / (1 + np.exp(-logits))
        y = (np.random.rand(n_samples) < probs).astype(np.float32)
    
    return X, y


@app.function(gpu="A100", image=image, timeout=1800)
def benchmark_gam_vs_ebm(
    n_samples: int = 100_000,
    n_features: int = 20,
    n_rounds: int = 500,
):
    """Compare OpenBoost GPU-GAM vs InterpretML EBM.
    
    Args:
        n_samples: Number of training samples
        n_features: Number of features
        n_rounds: Number of boosting rounds (for fair comparison)
        
    Returns:
        Benchmark results dict
    """
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import time
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("OpenBoost GPU-GAM vs InterpretML EBM Benchmark")
    print("=" * 60)
    
    # Check GPU
    try:
        from numba import cuda
        print(f"GPU: {cuda.get_current_device().name}")
    except Exception as e:
        print(f"GPU not available: {e}")
    
    # Generate data
    print(f"\nGenerating data: {n_samples:,} samples × {n_features} features")
    X, y = generate_data(n_samples, n_features, task="regression")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = {
        "n_samples": n_samples,
        "n_features": n_features,
        "n_rounds": n_rounds,
    }
    
    # =========================================================================
    # 1. OpenBoost GPU-GAM
    # =========================================================================
    print("\n" + "-" * 40)
    print("1. OpenBoost GPU-GAM")
    print("-" * 40)
    
    try:
        import openboost as ob
        from openboost._gam import OpenBoostGAM
        
        print(f"Backend: {ob.get_backend()}")
        
        gam = OpenBoostGAM(
            n_rounds=n_rounds,
            learning_rate=0.05,
            reg_lambda=1.0,
            loss='mse',
        )
        
        # Warmup (JIT compilation)
        gam_warmup = OpenBoostGAM(n_rounds=10, learning_rate=0.1)
        gam_warmup.fit(X_train[:1000], y_train[:1000])
        cuda.synchronize()
        
        # Benchmark training
        start = time.perf_counter()
        gam.fit(X_train, y_train)
        cuda.synchronize()
        ob_train_time = time.perf_counter() - start
        
        # Benchmark inference
        start = time.perf_counter()
        y_pred_ob = gam.predict(X_test)
        cuda.synchronize()
        ob_pred_time = time.perf_counter() - start
        
        ob_mse = mean_squared_error(y_test, y_pred_ob)
        ob_r2 = r2_score(y_test, y_pred_ob)
        
        print(f"Train time:    {ob_train_time:.3f}s")
        print(f"Predict time:  {ob_pred_time*1000:.2f}ms")
        print(f"MSE:           {ob_mse:.6f}")
        print(f"R²:            {ob_r2:.4f}")
        
        results["openboost_gam"] = {
            "train_time_s": ob_train_time,
            "predict_time_ms": ob_pred_time * 1000,
            "mse": ob_mse,
            "r2": ob_r2,
        }
        
    except Exception as e:
        print(f"OpenBoost GAM failed: {e}")
        import traceback
        traceback.print_exc()
        results["openboost_gam"] = {"error": str(e)}
    
    # =========================================================================
    # 2. InterpretML EBM
    # =========================================================================
    print("\n" + "-" * 40)
    print("2. InterpretML EBM")
    print("-" * 40)
    
    try:
        from interpret.glassbox import ExplainableBoostingRegressor
        
        # EBM with comparable settings
        # Note: EBM's "outer_bags" and "inner_bags" add bagging overhead
        # For fair comparison, we disable some features
        ebm = ExplainableBoostingRegressor(
            max_rounds=n_rounds,
            learning_rate=0.05,
            min_samples_leaf=2,
            max_bins=256,
            outer_bags=1,  # Disable bagging for speed comparison
            inner_bags=0,
            interactions=0,  # No pairwise interactions (pure GAM)
            n_jobs=-1,  # Use all CPU cores
        )
        
        # Benchmark training
        start = time.perf_counter()
        ebm.fit(X_train, y_train)
        ebm_train_time = time.perf_counter() - start
        
        # Benchmark inference
        start = time.perf_counter()
        y_pred_ebm = ebm.predict(X_test)
        ebm_pred_time = time.perf_counter() - start
        
        ebm_mse = mean_squared_error(y_test, y_pred_ebm)
        ebm_r2 = r2_score(y_test, y_pred_ebm)
        
        print(f"Train time:    {ebm_train_time:.3f}s")
        print(f"Predict time:  {ebm_pred_time*1000:.2f}ms")
        print(f"MSE:           {ebm_mse:.6f}")
        print(f"R²:            {ebm_r2:.4f}")
        
        results["interpretml_ebm"] = {
            "train_time_s": ebm_train_time,
            "predict_time_ms": ebm_pred_time * 1000,
            "mse": ebm_mse,
            "r2": ebm_r2,
        }
        
    except Exception as e:
        print(f"InterpretML EBM failed: {e}")
        import traceback
        traceback.print_exc()
        results["interpretml_ebm"] = {"error": str(e)}
    
    # =========================================================================
    # 3. XGBoost (baseline, non-interpretable)
    # =========================================================================
    print("\n" + "-" * 40)
    print("3. XGBoost (baseline)")
    print("-" * 40)
    
    try:
        import xgboost as xgb
        
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_rounds,
            learning_rate=0.05,
            max_depth=6,
            tree_method="hist",
            device="cuda",
            n_jobs=-1,
        )
        
        # Benchmark training
        start = time.perf_counter()
        xgb_model.fit(X_train, y_train)
        cuda.synchronize()
        xgb_train_time = time.perf_counter() - start
        
        # Benchmark inference
        start = time.perf_counter()
        y_pred_xgb = xgb_model.predict(X_test)
        cuda.synchronize()
        xgb_pred_time = time.perf_counter() - start
        
        xgb_mse = mean_squared_error(y_test, y_pred_xgb)
        xgb_r2 = r2_score(y_test, y_pred_xgb)
        
        print(f"Train time:    {xgb_train_time:.3f}s")
        print(f"Predict time:  {xgb_pred_time*1000:.2f}ms")
        print(f"MSE:           {xgb_mse:.6f}")
        print(f"R²:            {xgb_r2:.4f}")
        
        results["xgboost"] = {
            "train_time_s": xgb_train_time,
            "predict_time_ms": xgb_pred_time * 1000,
            "mse": xgb_mse,
            "r2": xgb_r2,
        }
        
    except Exception as e:
        print(f"XGBoost failed: {e}")
        results["xgboost"] = {"error": str(e)}
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    def safe_get(d, key, default="N/A"):
        if "error" in d:
            return default
        return d.get(key, default)
    
    print(f"\n{'Model':<25} {'Train (s)':<12} {'Predict (ms)':<14} {'R²':<10}")
    print("-" * 60)
    
    for name, key in [
        ("OpenBoost GPU-GAM", "openboost_gam"),
        ("InterpretML EBM", "interpretml_ebm"),
        ("XGBoost (GPU)", "xgboost"),
    ]:
        d = results.get(key, {})
        train = safe_get(d, "train_time_s")
        pred = safe_get(d, "predict_time_ms")
        r2 = safe_get(d, "r2")
        
        train_str = f"{train:.3f}" if isinstance(train, float) else train
        pred_str = f"{pred:.2f}" if isinstance(pred, float) else pred
        r2_str = f"{r2:.4f}" if isinstance(r2, float) else r2
        
        print(f"{name:<25} {train_str:<12} {pred_str:<14} {r2_str:<10}")
    
    # Speedup calculation
    if ("openboost_gam" in results and "interpretml_ebm" in results and 
        "error" not in results["openboost_gam"] and "error" not in results["interpretml_ebm"]):
        speedup = results["interpretml_ebm"]["train_time_s"] / results["openboost_gam"]["train_time_s"]
        print(f"\nOpenBoost GPU-GAM is {speedup:.1f}x faster than InterpretML EBM")
        results["speedup_vs_ebm"] = speedup
    
    return results


@app.function(gpu="A100", image=image, timeout=3600)
def benchmark_scaling(max_samples: int = 1_000_000):
    """Benchmark how both scale with data size."""
    import sys
    sys.path.insert(0, "/root")
    
    import numpy as np
    import time
    from numba import cuda
    
    print("=" * 60)
    print("Scaling Benchmark: OpenBoost GPU-GAM vs InterpretML EBM")
    print("=" * 60)
    print(f"GPU: {cuda.get_current_device().name}")
    
    import openboost as ob
    from openboost._gam import OpenBoostGAM
    from interpret.glassbox import ExplainableBoostingRegressor
    
    n_features = 20
    n_rounds = 200
    
    # Warmup
    X_warm, y_warm = generate_data(1000, n_features)
    OpenBoostGAM(n_rounds=10).fit(X_warm, y_warm)
    cuda.synchronize()
    
    results = []
    
    for n_samples in [10_000, 50_000, 100_000, 500_000, max_samples]:
        if n_samples > max_samples:
            break
            
        print(f"\n--- {n_samples:,} samples ---")
        
        X, y = generate_data(n_samples, n_features)
        row = {"n_samples": n_samples}
        
        # OpenBoost GPU-GAM
        try:
            gam = OpenBoostGAM(n_rounds=n_rounds, learning_rate=0.05)
            start = time.perf_counter()
            gam.fit(X, y)
            cuda.synchronize()
            row["openboost_time"] = time.perf_counter() - start
            print(f"  OpenBoost GPU-GAM: {row['openboost_time']:.2f}s")
        except Exception as e:
            print(f"  OpenBoost failed: {e}")
            row["openboost_time"] = None
        
        # InterpretML EBM (only for smaller sizes due to time)
        if n_samples <= 100_000:
            try:
                ebm = ExplainableBoostingRegressor(
                    max_rounds=n_rounds,
                    learning_rate=0.05,
                    outer_bags=1,
                    inner_bags=0,
                    interactions=0,
                    n_jobs=-1,
                )
                start = time.perf_counter()
                ebm.fit(X, y)
                row["ebm_time"] = time.perf_counter() - start
                print(f"  InterpretML EBM:   {row['ebm_time']:.2f}s")
            except Exception as e:
                print(f"  EBM failed: {e}")
                row["ebm_time"] = None
        else:
            print(f"  InterpretML EBM:   (skipped - too slow)")
            row["ebm_time"] = None
        
        if row.get("openboost_time") and row.get("ebm_time"):
            row["speedup"] = row["ebm_time"] / row["openboost_time"]
            print(f"  Speedup: {row['speedup']:.1f}x")
        
        results.append(row)
    
    print("\n" + "=" * 60)
    print("SCALING SUMMARY")
    print("=" * 60)
    print(f"\n{'Samples':<12} {'OpenBoost (s)':<15} {'EBM (s)':<12} {'Speedup':<10}")
    print("-" * 50)
    for r in results:
        ob_str = f"{r['openboost_time']:.2f}" if r.get('openboost_time') else "N/A"
        ebm_str = f"{r['ebm_time']:.2f}" if r.get('ebm_time') else "N/A"
        sp_str = f"{r['speedup']:.1f}x" if r.get('speedup') else "N/A"
        print(f"{r['n_samples']:<12,} {ob_str:<15} {ebm_str:<12} {sp_str:<10}")
    
    return results


@app.local_entrypoint()
def main():
    """Run benchmarks on Modal."""
    print("Running GAM vs EBM benchmark on Modal A100...")
    
    # Main benchmark
    results = benchmark_gam_vs_ebm.remote(
        n_samples=100_000,
        n_features=20,
        n_rounds=500,
    )
    
    print("\n\nFinal Results:")
    print(results)
    
    # Scaling benchmark (optional, takes longer)
    # scaling_results = benchmark_scaling.remote(max_samples=500_000)
    # print("\n\nScaling Results:")
    # print(scaling_results)


# For local execution without Modal
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        # Run locally
        print("Running locally...")
        
        import numpy as np
        import time
        from sklearn.metrics import mean_squared_error, r2_score
        from sklearn.model_selection import train_test_split
        
        # Add openboost to path
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        
        n_samples = 50_000
        n_features = 20
        n_rounds = 200
        
        # Generate data
        X, y = generate_data(n_samples, n_features)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print(f"Data: {n_samples:,} samples × {n_features} features")
        
        # OpenBoost GPU-GAM
        try:
            import openboost as ob
            from openboost._gam import OpenBoostGAM
            
            print(f"\nOpenBoost backend: {ob.get_backend()}")
            
            gam = OpenBoostGAM(n_rounds=n_rounds, learning_rate=0.05)
            start = time.perf_counter()
            gam.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            
            y_pred = gam.predict(X_test)
            print(f"OpenBoost GPU-GAM: {train_time:.2f}s, R²={r2_score(y_test, y_pred):.4f}")
        except Exception as e:
            print(f"OpenBoost failed: {e}")
            import traceback
            traceback.print_exc()
        
        # InterpretML EBM
        try:
            from interpret.glassbox import ExplainableBoostingRegressor
            
            ebm = ExplainableBoostingRegressor(
                max_rounds=n_rounds,
                learning_rate=0.05,
                outer_bags=1,
                inner_bags=0,
                interactions=0,
                n_jobs=-1,
            )
            start = time.perf_counter()
            ebm.fit(X_train, y_train)
            train_time = time.perf_counter() - start
            
            y_pred = ebm.predict(X_test)
            print(f"InterpretML EBM:   {train_time:.2f}s, R²={r2_score(y_test, y_pred):.4f}")
        except ImportError:
            print("InterpretML not installed. Run: pip install interpret")
        except Exception as e:
            print(f"EBM failed: {e}")
    else:
        print("Usage:")
        print("  Modal:  uv run modal run benchmarks/ebm_benchmark.py")
        print("  Local:  uv run python benchmarks/ebm_benchmark.py --local")

