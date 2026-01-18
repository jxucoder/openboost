"""Test openboost on GPU via Modal."""

import modal

app = modal.App("openboost-test")

# Use CUDA devel image (includes NVVM compiler needed by numba)
image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install("numpy>=1.24", "numba-cuda>=0.23", "xgboost>=2.0", "scikit-learn")
    .add_local_dir("src", "/root/src")
)


@app.function(gpu="T4", image=image, timeout=180)
def test_openboost():
    """Run MVP success criteria test with XGBoost comparison."""
    import time
    import numpy as np
    import xgboost as xgb
    
    # Add mounted source to path
    import sys
    sys.path.insert(0, "/root/src")
    
    import openboost as ob
    
    print("OpenBoost version:", ob.__version__)
    print("XGBoost version:", xgb.__version__)
    
    # Generate test data (from MVP success criteria)
    np.random.seed(42)
    X = np.random.randn(100_000, 50).astype(np.float32)
    y = np.random.randn(100_000).astype(np.float32)
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    
    # Matching config
    n_trees = 20
    max_depth = 6
    learning_rate = 0.1
    
    # --- OpenBoost ---
    print("\n" + "="*50)
    print("OpenBoost")
    print("="*50)
    
    start = time.time()
    ob_model = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
    )
    ob_model.fit(X, y)
    ob_train_time = time.time() - start
    print(f"Training time: {ob_train_time:.2f}s")
    
    start = time.time()
    ob_pred = ob_model.predict(X)
    ob_predict_time = time.time() - start
    print(f"Prediction time: {ob_predict_time:.2f}s")
    
    ob_mse = float(np.mean((ob_pred - y) ** 2))
    print(f"MSE: {ob_mse:.4f}")
    
    # --- XGBoost GPU ---
    print("\n" + "="*50)
    print("XGBoost (GPU)")
    print("="*50)
    
    start = time.time()
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=learning_rate,
        tree_method="hist",
        device="cuda",
        objective="reg:squarederror",
    )
    xgb_model.fit(X, y)
    xgb_train_time = time.time() - start
    print(f"Training time: {xgb_train_time:.2f}s")
    
    start = time.time()
    xgb_pred = xgb_model.predict(X)
    xgb_predict_time = time.time() - start
    print(f"Prediction time: {xgb_predict_time:.2f}s")
    
    xgb_mse = float(np.mean((xgb_pred - y) ** 2))
    print(f"MSE: {xgb_mse:.4f}")
    
    # --- Comparison ---
    print("\n" + "="*50)
    print("Comparison")
    print("="*50)
    print(f"{'Metric':<20} {'OpenBoost':>12} {'XGBoost':>12} {'Ratio':>10}")
    print("-"*54)
    print(f"{'Train time (s)':<20} {ob_train_time:>12.2f} {xgb_train_time:>12.2f} {ob_train_time/xgb_train_time:>10.1f}x")
    print(f"{'Predict time (s)':<20} {ob_predict_time:>12.2f} {xgb_predict_time:>12.2f} {ob_predict_time/xgb_predict_time:>10.1f}x")
    print(f"{'MSE':<20} {ob_mse:>12.4f} {xgb_mse:>12.4f}")
    
    return {
        "openboost": {
            "train_time": ob_train_time,
            "predict_time": ob_predict_time,
            "mse": ob_mse,
        },
        "xgboost": {
            "train_time": xgb_train_time,
            "predict_time": xgb_predict_time,
            "mse": xgb_mse,
        },
    }


@app.local_entrypoint()
def main():
    print("Running openboost vs xgboost benchmark on GPU...")
    result = test_openboost.remote()
    
    ob = result["openboost"]
    xgb = result["xgboost"]
    
    print("\n=== Final Results ===")
    print(f"{'Metric':<20} {'OpenBoost':>12} {'XGBoost':>12}")
    print("-"*44)
    print(f"{'Train time (s)':<20} {ob['train_time']:>12.2f} {xgb['train_time']:>12.2f}")
    print(f"{'Predict time (s)':<20} {ob['predict_time']:>12.2f} {xgb['predict_time']:>12.2f}")
    print(f"{'MSE':<20} {ob['mse']:>12.4f} {xgb['mse']:>12.4f}")
