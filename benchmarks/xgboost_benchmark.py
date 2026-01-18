"""Benchmark: OpenBoost vs XGBoost on Multiple Tasks.

Run locally:
    uv run python benchmarks/xgboost_benchmark.py --local

Run on Modal (cloud A100):
    uv run modal run benchmarks/xgboost_benchmark.py
"""

import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

app = modal.App("openboost-xgboost-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "scikit-learn>=1.0",
        "xgboost>=2.0",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost",
    )
)


# =============================================================================
# Data Generators
# =============================================================================

def generate_regression_data(n_samples: int, n_features: int, noise: float = 0.1):
    """Generate regression data."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Non-linear target with interactions
    y = (
        np.sin(X[:, 0] * 2) + 
        0.5 * X[:, 1] ** 2 + 
        0.3 * X[:, 2] * X[:, 3] +
        noise * np.random.randn(n_samples)
    ).astype(np.float32)
    return X, y


def generate_binary_data(n_samples: int, n_features: int):
    """Generate binary classification data."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + 0.2 * X[:, 3]
    probs = 1 / (1 + np.exp(-logits))
    y = (np.random.rand(n_samples) < probs).astype(np.float32)
    return X, y


def generate_multiclass_data(n_samples: int, n_features: int, n_classes: int = 5):
    """Generate multi-class classification data."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Create class boundaries based on first few features
    scores = np.zeros((n_samples, n_classes))
    for k in range(n_classes):
        scores[:, k] = X[:, k % n_features] + 0.5 * X[:, (k + 1) % n_features]
    y = np.argmax(scores + 0.5 * np.random.randn(n_samples, n_classes), axis=1)
    return X, y.astype(np.int32)


def generate_quantile_data(n_samples: int, n_features: int):
    """Generate heteroscedastic data for quantile regression."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Heteroscedastic noise: variance depends on X[:, 0]
    noise_std = 0.5 + np.abs(X[:, 0])
    y = (X[:, 0] + 0.5 * X[:, 1] + noise_std * np.random.randn(n_samples)).astype(np.float32)
    return X, y


def generate_poisson_data(n_samples: int, n_features: int):
    """Generate count data for Poisson regression."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Log-linear model
    log_mu = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    mu = np.exp(np.clip(log_mu, -5, 5))
    y = np.random.poisson(mu).astype(np.float32)
    return X, y


def generate_gamma_data(n_samples: int, n_features: int):
    """Generate positive continuous data for Gamma regression."""
    import numpy as np
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    # Log-linear model for mean
    log_mu = 2.0 + 0.3 * X[:, 0] + 0.2 * X[:, 1]
    mu = np.exp(np.clip(log_mu, -3, 5))
    # Gamma with shape=2
    shape = 2.0
    scale = mu / shape
    y = np.random.gamma(shape, scale).astype(np.float32)
    return X, y


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_regression(X_train, X_test, y_train, y_test, n_trees=100, max_depth=6, use_gpu=False):
    """Benchmark regression task."""
    import numpy as np
    import time
    from sklearn.metrics import mean_squared_error, r2_score
    
    results = {}
    
    # OpenBoost
    import openboost as ob
    model = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        loss='mse',
    )
    
    # Warmup
    ob.GradientBoosting(n_trees=5, max_depth=3).fit(X_train[:1000], y_train[:1000])
    if use_gpu:
        from numba import cuda
        cuda.synchronize()
    
    start = time.perf_counter()
    model.fit(X_train, y_train)
    if use_gpu:
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    results['openboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    
    # XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda' if use_gpu else 'cpu',
    )
    
    start = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    if use_gpu:
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_xgb = xgb_model.predict(X_test)
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    results['xgboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'r2': r2_score(y_test, y_pred_xgb),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
    }
    
    return results


def benchmark_binary(X_train, X_test, y_train, y_test, n_trees=100, max_depth=6, use_gpu=False):
    """Benchmark binary classification task."""
    import numpy as np
    import time
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    results = {}
    
    # OpenBoost
    import openboost as ob
    model = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        loss='logloss',
    )
    
    start = time.perf_counter()
    model.fit(X_train, y_train)
    if use_gpu:
        from numba import cuda
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_raw = model.predict(X_test)
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    # Convert to probabilities
    y_pred_prob = 1 / (1 + np.exp(-y_pred_raw))
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    results['openboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'auc': roc_auc_score(y_test, y_pred_prob),
        'accuracy': accuracy_score(y_test, y_pred),
    }
    
    # XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda' if use_gpu else 'cpu',
    )
    
    start = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    if use_gpu:
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    y_pred_xgb = (y_pred_xgb_prob > 0.5).astype(int)
    
    results['xgboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'auc': roc_auc_score(y_test, y_pred_xgb_prob),
        'accuracy': accuracy_score(y_test, y_pred_xgb),
    }
    
    return results


def benchmark_multiclass(X_train, X_test, y_train, y_test, n_classes=5, n_trees=100, max_depth=6, use_gpu=False):
    """Benchmark multi-class classification task."""
    import numpy as np
    import time
    from sklearn.metrics import accuracy_score, log_loss
    
    results = {}
    
    # OpenBoost MultiClass
    import openboost as ob
    model = ob.MultiClassGradientBoosting(
        n_classes=n_classes,
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
    )
    
    start = time.perf_counter()
    model.fit(X_train, y_train)
    if use_gpu:
        from numba import cuda
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_prob = model.predict_proba(X_test)
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    results['openboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'accuracy': accuracy_score(y_test, y_pred),
        'logloss': log_loss(y_test, y_pred_prob),
    }
    
    # XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda' if use_gpu else 'cpu',
        objective='multi:softprob',
        num_class=n_classes,
    )
    
    start = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    if use_gpu:
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_xgb_prob = xgb_model.predict_proba(X_test)
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    y_pred_xgb = np.argmax(y_pred_xgb_prob, axis=1)
    
    results['xgboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'logloss': log_loss(y_test, y_pred_xgb_prob),
    }
    
    return results


def benchmark_poisson(X_train, X_test, y_train, y_test, n_trees=100, max_depth=6, use_gpu=False):
    """Benchmark Poisson regression task."""
    import numpy as np
    import time
    
    def poisson_deviance(y_true, y_pred):
        """Compute Poisson deviance."""
        y_pred = np.maximum(y_pred, 1e-8)
        return 2 * np.mean(y_pred - y_true + y_true * np.log(np.maximum(y_true, 1e-8) / y_pred))
    
    results = {}
    
    # OpenBoost
    import openboost as ob
    model = ob.GradientBoosting(
        n_trees=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        loss='poisson',
    )
    
    start = time.perf_counter()
    model.fit(X_train, y_train)
    if use_gpu:
        from numba import cuda
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_raw = model.predict(X_test)
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    # Poisson uses log link
    y_pred = np.exp(y_pred_raw)
    
    results['openboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'deviance': poisson_deviance(y_test, y_pred),
        'mean_pred': np.mean(y_pred),
    }
    
    # XGBoost
    import xgboost as xgb
    xgb_model = xgb.XGBRegressor(
        n_estimators=n_trees,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method='hist',
        device='cuda' if use_gpu else 'cpu',
        objective='count:poisson',
    )
    
    start = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    if use_gpu:
        cuda.synchronize()
    train_time = time.perf_counter() - start
    
    start = time.perf_counter()
    y_pred_xgb = xgb_model.predict(X_test)
    if use_gpu:
        cuda.synchronize()
    pred_time = time.perf_counter() - start
    
    results['xgboost'] = {
        'train_time': train_time,
        'pred_time': pred_time * 1000,
        'deviance': poisson_deviance(y_test, y_pred_xgb),
        'mean_pred': np.mean(y_pred_xgb),
    }
    
    return results


# =============================================================================
# Main Benchmark Runner
# =============================================================================

def print_results(task_name, results, metric1_name, metric2_name):
    """Print formatted results."""
    print(f"\n{'─' * 60}")
    print(f"Task: {task_name}")
    print(f"{'─' * 60}")
    print(f"{'Model':<15} {'Train (s)':<12} {'Pred (ms)':<12} {metric1_name:<12} {metric2_name:<12}")
    print(f"{'─' * 60}")
    
    for name in ['openboost', 'xgboost']:
        r = results[name]
        m1 = list(r.values())[2]  # First metric after times
        m2 = list(r.values())[3]  # Second metric
        print(f"{name:<15} {r['train_time']:<12.3f} {r['pred_time']:<12.2f} {m1:<12.4f} {m2:<12.4f}")
    
    # Speedup
    speedup = results['xgboost']['train_time'] / results['openboost']['train_time']
    print(f"{'─' * 60}")
    print(f"Speedup: {speedup:.2f}x {'(OpenBoost faster)' if speedup > 1 else '(XGBoost faster)'}")


def run_all_benchmarks(n_samples=50_000, n_features=20, n_trees=100, max_depth=6, use_gpu=False):
    """Run all benchmark tasks."""
    from sklearn.model_selection import train_test_split
    
    print("=" * 60)
    print("OPENBOOST vs XGBOOST BENCHMARK")
    print("=" * 60)
    print(f"Config: {n_samples:,} samples, {n_features} features, {n_trees} trees, depth {max_depth}")
    print(f"Device: {'GPU' if use_gpu else 'CPU'}")
    
    all_results = {}
    
    # 1. Regression
    print("\n[1/5] Regression (MSE)...")
    X, y = generate_regression_data(n_samples, n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = benchmark_regression(X_train, X_test, y_train, y_test, n_trees, max_depth, use_gpu)
    print_results("Regression (MSE)", results, "R²", "RMSE")
    all_results['regression'] = results
    
    # 2. Binary Classification
    print("\n[2/5] Binary Classification (LogLoss)...")
    X, y = generate_binary_data(n_samples, n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = benchmark_binary(X_train, X_test, y_train, y_test, n_trees, max_depth, use_gpu)
    print_results("Binary Classification", results, "AUC", "Accuracy")
    all_results['binary'] = results
    
    # 3. Multi-class Classification
    print("\n[3/5] Multi-class Classification (Softmax)...")
    n_classes = 5
    X, y = generate_multiclass_data(n_samples, n_features, n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = benchmark_multiclass(X_train, X_test, y_train, y_test, n_classes, n_trees, max_depth, use_gpu)
    print_results("Multi-class (5 classes)", results, "Accuracy", "LogLoss")
    all_results['multiclass'] = results
    
    # 4. Poisson Regression
    print("\n[4/5] Poisson Regression...")
    X, y = generate_poisson_data(n_samples, n_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = benchmark_poisson(X_train, X_test, y_train, y_test, n_trees, max_depth, use_gpu)
    print_results("Poisson Regression", results, "Deviance", "Mean Pred")
    all_results['poisson'] = results
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Task':<25} {'OpenBoost (s)':<15} {'XGBoost (s)':<15} {'Speedup':<10}")
    print("─" * 65)
    
    for task, res in all_results.items():
        ob_time = res['openboost']['train_time']
        xgb_time = res['xgboost']['train_time']
        speedup = xgb_time / ob_time
        faster = "OB" if speedup > 1 else "XGB"
        print(f"{task:<25} {ob_time:<15.3f} {xgb_time:<15.3f} {speedup:.2f}x ({faster})")
    
    return all_results


# =============================================================================
# Modal Entry Points
# =============================================================================

@app.function(gpu="A100", image=image, timeout=1800)
def benchmark_gpu(n_samples: int = 100_000, n_features: int = 20, n_trees: int = 100):
    """Run benchmark on GPU."""
    import sys
    sys.path.insert(0, "/root")
    
    from numba import cuda
    print(f"GPU: {cuda.get_current_device().name}")
    
    return run_all_benchmarks(
        n_samples=n_samples,
        n_features=n_features,
        n_trees=n_trees,
        max_depth=6,
        use_gpu=True,
    )


@app.local_entrypoint()
def main():
    """Run benchmark on Modal."""
    print("Running OpenBoost vs XGBoost benchmark on Modal A100...")
    results = benchmark_gpu.remote(n_samples=100_000, n_features=20, n_trees=100)
    print("\n\nFinal Results:")
    print(results)


# =============================================================================
# Local Execution
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        print("Running locally on CPU...")
        
        import sys as system
        system.path.insert(0, str(PROJECT_ROOT / "src"))
        
        run_all_benchmarks(
            n_samples=20_000,  # Smaller for CPU
            n_features=20,
            n_trees=50,
            max_depth=6,
            use_gpu=False,
        )
    else:
        print("Usage:")
        print("  Modal:  uv run modal run benchmarks/xgboost_benchmark.py")
        print("  Local:  uv run python benchmarks/xgboost_benchmark.py --local")
