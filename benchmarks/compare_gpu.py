"""GPU benchmarks: OpenBoost vs competitors on Modal A100.

Three comparisons in one file:
  1. GradientBoosting vs XGBoost  — regression, binary, multiclass, poisson
  2. NaturalBoost vs NGBoost      — distributional GBDT (uncertainty)
  3. OpenBoostGAM vs InterpretML  — interpretable models (GAM)

Usage:
    # Run everything on Modal A100
    uv run modal run benchmarks/compare_gpu.py

    # Run a single comparison
    uv run modal run benchmarks/compare_gpu.py --bench xgboost
    uv run modal run benchmarks/compare_gpu.py --bench ngboost
    uv run modal run benchmarks/compare_gpu.py --bench ebm

    # Run locally (CPU, smaller data)
    uv run python benchmarks/compare_gpu.py --local
    uv run python benchmarks/compare_gpu.py --local --bench ngboost
"""

from __future__ import annotations

import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

try:
    import modal

    app = modal.App("openboost-gpu-bench")

    image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
        )
        .pip_install(
            "numpy>=1.24",
            "numba>=0.60",
            "scikit-learn>=1.0",
            "xgboost>=2.0",
            "ngboost>=0.5",
            "interpret>=0.6",
        )
        .add_local_dir(
            str(PROJECT_ROOT / "src" / "openboost"),
            remote_path="/root/openboost",
        )
    )
except ImportError:
    modal = None
    app = None
    image = None


# =============================================================================
# Data generators
# =============================================================================


def _generate_regression(n_samples, n_features=20, seed=42):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (
        np.sin(X[:, 0] * 2)
        + 0.5 * X[:, 1] ** 2
        + 0.3 * X[:, 2] * X[:, 3]
        + 0.1 * rng.randn(n_samples)
    ).astype(np.float32)
    return X, y


def _generate_binary(n_samples, n_features=20, seed=42):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    y = (rng.rand(n_samples) < 1 / (1 + np.exp(-logits))).astype(np.float32)
    return X, y


def _generate_multiclass(n_samples, n_features=20, n_classes=5, seed=42):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    scores = np.zeros((n_samples, n_classes))
    for k in range(n_classes):
        scores[:, k] = X[:, k % n_features] + 0.5 * X[:, (k + 1) % n_features]
    y = np.argmax(scores + 0.5 * rng.randn(n_samples, n_classes), axis=1)
    return X, y.astype(np.int32)


def _generate_poisson(n_samples, n_features=20, seed=42):
    import numpy as np

    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    log_mu = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    mu = np.exp(np.clip(log_mu, -5, 5))
    y = rng.poisson(mu).astype(np.float32)
    return X, y


def _train_test_split(X, y, test_size=0.2, seed=42):
    import numpy as np

    rng = np.random.RandomState(seed)
    n = len(y)
    idx = rng.permutation(n)
    split = int(n * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]


# =============================================================================
# Timing helper
# =============================================================================


def _time_fit(create_model_fn, X, y, n_trials=3, sync_gpu=False):
    """Time model.fit() over n_trials, return (median_time, last_fitted_model)."""
    times = []
    model = None
    for _ in range(n_trials):
        model = create_model_fn()
        t0 = time.perf_counter()
        model.fit(X, y)
        if sync_gpu:
            from numba import cuda
            cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], model


# =============================================================================
# Benchmark 1: GradientBoosting vs XGBoost
# =============================================================================


def bench_xgboost(n_samples=50_000, n_trees=100, max_depth=6, use_gpu=False):
    """Compare OpenBoost vs XGBoost across tasks."""
    import numpy as np
    from sklearn.metrics import accuracy_score, r2_score

    import openboost as ob

    sync = use_gpu

    # Warmup JIT — two iterations to ensure all CUDA kernels are compiled
    X_w, y_w = _generate_regression(500)
    for _ in range(2):
        ob.GradientBoosting(n_trees=3, max_depth=max_depth).fit(X_w, y_w)
    if sync:
        from numba import cuda
        cuda.synchronize()

    xgb_device = "cuda" if use_gpu else "cpu"

    results = {}

    tasks = [
        ("regression", _generate_regression, "mse", "R²"),
        ("binary", _generate_binary, "logloss", "AUC"),
        ("multiclass", _generate_multiclass, "softmax", "Accuracy"),
        ("poisson", _generate_poisson, "poisson", "Deviance"),
    ]

    for task_name, gen_fn, ob_loss, metric_label in tasks:
        X, y = gen_fn(n_samples)
        X_train, X_test, y_train, y_test = _train_test_split(X, y)

        # --- OpenBoost ---
        if task_name == "multiclass":
            n_classes = len(np.unique(y))
            ob_time, ob_model = _time_fit(
                lambda nc=n_classes: ob.MultiClassGradientBoosting(
                    n_classes=nc, n_trees=n_trees, max_depth=max_depth,
                    learning_rate=0.1,
                ),
                X_train, y_train, sync_gpu=sync,
            )
            ob_pred = np.argmax(ob_model.predict_proba(X_test), axis=1)
        else:
            ob_time, ob_model = _time_fit(
                lambda loss=ob_loss: ob.GradientBoosting(
                    n_trees=n_trees, max_depth=max_depth, learning_rate=0.1,
                    loss=loss,
                ),
                X_train, y_train, sync_gpu=sync,
            )
            ob_pred = ob_model.predict(X_test)

        # --- XGBoost ---
        import xgboost as xgb

        if task_name == "regression":
            xgb_time, xgb_model = _time_fit(
                lambda: xgb.XGBRegressor(
                    n_estimators=n_trees, max_depth=max_depth, learning_rate=0.1,
                    tree_method="hist", device=xgb_device, verbosity=0,
                ),
                X_train, y_train,
            )
            xgb_pred = xgb_model.predict(X_test)
        elif task_name == "poisson":
            xgb_time, xgb_model = _time_fit(
                lambda: xgb.XGBRegressor(
                    n_estimators=n_trees, max_depth=max_depth, learning_rate=0.1,
                    tree_method="hist", device=xgb_device, objective="count:poisson",
                    verbosity=0,
                ),
                X_train, y_train,
            )
            xgb_pred = xgb_model.predict(X_test)
        else:
            xgb_time, xgb_model = _time_fit(
                lambda: xgb.XGBClassifier(
                    n_estimators=n_trees, max_depth=max_depth, learning_rate=0.1,
                    tree_method="hist", device=xgb_device, verbosity=0,
                ),
                X_train, y_train,
            )
            xgb_pred = xgb_model.predict(X_test)

        # --- Metrics ---
        if task_name == "regression":
            ob_metric = r2_score(y_test, ob_pred)
            xgb_metric = r2_score(y_test, xgb_pred)
        elif task_name == "poisson":
            ob_exp = np.exp(ob_pred)
            ob_metric = float(np.mean(ob_exp - y_test * np.log(np.maximum(ob_exp, 1e-8))))
            xgb_metric = float(np.mean(xgb_pred - y_test * np.log(np.maximum(xgb_pred, 1e-8))))
        elif task_name == "binary":
            ob_labels = (ob_pred > 0).astype(float) if np.any(ob_pred < 0) else ob_pred
            ob_metric = accuracy_score(y_test, ob_labels)
            xgb_metric = accuracy_score(y_test, xgb_pred)
        else:
            ob_metric = accuracy_score(y_test, ob_pred)
            xgb_metric = accuracy_score(y_test, xgb_pred)

        speedup = xgb_time / ob_time
        results[task_name] = {
            "ob_time": ob_time, "xgb_time": xgb_time, "speedup": speedup,
            "ob_metric": float(ob_metric), "xgb_metric": float(xgb_metric),
            "metric_label": metric_label,
        }

    # Print results
    print(f"\n{'='*70}")
    print(f"  OpenBoost vs XGBoost  |  {n_samples:,} samples, {n_trees} trees, depth {max_depth}")
    print(f"  Device: {'GPU' if use_gpu else 'CPU'}")
    print(f"{'='*70}")
    print(f"  {'Task':<14} {'OB (s)':<10} {'XGB (s)':<10} {'Speedup':<10} {'OB metric':<12} {'XGB metric':<12}")
    print(f"  {'─'*66}")
    for task_name, r in results.items():
        faster = "OB" if r["speedup"] > 1 else "XGB"
        print(
            f"  {task_name:<14} {r['ob_time']:<10.3f} {r['xgb_time']:<10.3f} "
            f"{r['speedup']:.2f}x {faster:<4} {r['ob_metric']:<12.4f} {r['xgb_metric']:<12.4f}"
        )

    return results


# =============================================================================
# Benchmark 2: NaturalBoost vs NGBoost
# =============================================================================


def bench_ngboost(n_samples=10_000, n_trees=100, use_gpu=False):
    """Compare NaturalBoost vs NGBoost (distributional GBDT)."""
    import numpy as np
    from sklearn.datasets import fetch_california_housing

    import openboost as ob

    sync = use_gpu

    # Warmup
    X_w, y_w = _generate_regression(500)
    ob.NaturalBoostNormal(n_trees=3, max_depth=3, learning_rate=0.1).fit(X_w, y_w)
    if sync:
        from numba import cuda
        cuda.synchronize()

    results = {}

    # --- Synthetic data ---
    for n in [n_samples]:
        X, y = _generate_regression(n)
        X_train, X_test, y_train, y_test = _train_test_split(X, y)

        # NGBoost
        from ngboost import NGBRegressor
        from ngboost.distns import Normal

        ngb = NGBRegressor(Dist=Normal, n_estimators=n_trees, learning_rate=0.1, verbose=False)
        t0 = time.perf_counter()
        ngb.fit(X_train, y_train)
        ngb_time = time.perf_counter() - t0

        ngb_dist = ngb.pred_dist(X_test)
        ngb_nll = float(-ngb_dist.logpdf(y_test).mean())
        ngb_lower = ngb_dist.ppf(0.05)
        ngb_upper = ngb_dist.ppf(0.95)
        ngb_coverage = float(np.mean((y_test >= ngb_lower) & (y_test <= ngb_upper)))

        # NaturalBoost
        nb = ob.NaturalBoostNormal(n_trees=n_trees, learning_rate=0.1, max_depth=3)
        t0 = time.perf_counter()
        nb.fit(X_train, y_train)
        if sync:
            from numba import cuda
            cuda.synchronize()
        nb_time = time.perf_counter() - t0

        nb_nll = float(nb.nll(X_test, y_test))
        nb_lower, nb_upper = nb.predict_interval(X_test, alpha=0.1)
        nb_coverage = float(np.mean((y_test >= nb_lower) & (y_test <= nb_upper)))

        results[f"synthetic_{n}"] = {
            "ngb_time": ngb_time, "nb_time": nb_time,
            "speedup": ngb_time / nb_time,
            "ngb_nll": ngb_nll, "nb_nll": nb_nll,
            "ngb_coverage": ngb_coverage, "nb_coverage": nb_coverage,
        }

    # --- California Housing ---
    data = fetch_california_housing()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)
    X_train, X_test, y_train, y_test = _train_test_split(X, y)

    ngb = NGBRegressor(Dist=Normal, n_estimators=n_trees, learning_rate=0.1, verbose=False)
    t0 = time.perf_counter()
    ngb.fit(X_train, y_train)
    ngb_time = time.perf_counter() - t0
    ngb_nll = float(-ngb.pred_dist(X_test).logpdf(y_test).mean())

    nb = ob.NaturalBoostNormal(n_trees=n_trees, learning_rate=0.1, max_depth=3)
    t0 = time.perf_counter()
    nb.fit(X_train, y_train)
    if sync:
        from numba import cuda
        cuda.synchronize()
    nb_time = time.perf_counter() - t0
    nb_nll = float(nb.nll(X_test, y_test))

    results["california_housing"] = {
        "ngb_time": ngb_time, "nb_time": nb_time,
        "speedup": ngb_time / nb_time,
        "ngb_nll": ngb_nll, "nb_nll": nb_nll,
    }

    # Print
    print(f"\n{'='*70}")
    print(f"  NaturalBoost vs NGBoost  |  {n_trees} trees, Normal distribution")
    print(f"{'='*70}")
    print(f"  {'Dataset':<22} {'NGBoost (s)':<14} {'NatBoost (s)':<14} {'Speedup':<10} {'NGB NLL':<10} {'NB NLL':<10}")
    print(f"  {'─'*78}")
    for name, r in results.items():
        faster = "NB" if r["speedup"] > 1 else "NGB"
        print(
            f"  {name:<22} {r['ngb_time']:<14.2f} {r['nb_time']:<14.2f} "
            f"{r['speedup']:.2f}x {faster:<4} {r['ngb_nll']:<10.4f} {r['nb_nll']:<10.4f}"
        )

    return results


# =============================================================================
# Benchmark 3: OpenBoostGAM vs InterpretML EBM
# =============================================================================


def bench_ebm(n_samples=50_000, n_rounds=200, use_gpu=False):
    """Compare OpenBoostGAM vs InterpretML EBM."""
    from sklearn.metrics import r2_score

    from openboost import OpenBoostGAM

    sync = use_gpu

    # Warmup
    X_w, y_w = _generate_regression(500)
    OpenBoostGAM(n_rounds=10).fit(X_w, y_w)
    if sync:
        from numba import cuda
        cuda.synchronize()

    results = {}

    for n in [n_samples]:
        X, y = _generate_regression(n)
        X_train, X_test, y_train, y_test = _train_test_split(X, y)

        # OpenBoostGAM
        gam = OpenBoostGAM(n_rounds=n_rounds, learning_rate=0.05)
        t0 = time.perf_counter()
        gam.fit(X_train, y_train)
        if sync:
            from numba import cuda
            cuda.synchronize()
        gam_time = time.perf_counter() - t0
        gam_r2 = float(r2_score(y_test, gam.predict(X_test)))

        # InterpretML EBM
        from interpret.glassbox import ExplainableBoostingRegressor

        ebm = ExplainableBoostingRegressor(
            max_rounds=n_rounds, learning_rate=0.05,
            outer_bags=1, inner_bags=0, interactions=0, n_jobs=-1,
        )
        t0 = time.perf_counter()
        ebm.fit(X_train, y_train)
        ebm_time = time.perf_counter() - t0
        ebm_r2 = float(r2_score(y_test, ebm.predict(X_test)))

        results[f"synthetic_{n}"] = {
            "gam_time": gam_time, "ebm_time": ebm_time,
            "speedup": ebm_time / gam_time,
            "gam_r2": gam_r2, "ebm_r2": ebm_r2,
        }

    # Print
    print(f"\n{'='*70}")
    print(f"  OpenBoostGAM vs InterpretML EBM  |  {n_rounds} rounds")
    print(f"{'='*70}")
    print(f"  {'Dataset':<22} {'GAM (s)':<12} {'EBM (s)':<12} {'Speedup':<10} {'GAM R²':<10} {'EBM R²':<10}")
    print(f"  {'─'*74}")
    for name, r in results.items():
        print(
            f"  {name:<22} {r['gam_time']:<12.2f} {r['ebm_time']:<12.2f} "
            f"{r['speedup']:.1f}x       {r['gam_r2']:<10.4f} {r['ebm_r2']:<10.4f}"
        )

    return results


# =============================================================================
# Run all benchmarks
# =============================================================================


def run_all(use_gpu=False, bench=None, n_samples=None):
    """Run selected or all benchmarks."""
    import sys
    if use_gpu:
        sys.path.insert(0, "/root")

    import openboost as ob

    if use_gpu:
        ob.set_backend("cuda")

    print(f"OpenBoost backend: {ob.get_backend()}")
    if use_gpu:
        from numba import cuda
        gpu_name = cuda.get_current_device().name
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode()
        print(f"GPU: {gpu_name}")

    all_results = {}
    benches = [bench] if bench else ["xgboost", "ngboost", "ebm"]

    if "xgboost" in benches:
        n = n_samples or (50_000 if use_gpu else 20_000)
        all_results["xgboost"] = bench_xgboost(n_samples=n, use_gpu=use_gpu)

    if "ngboost" in benches:
        try:
            import ngboost  # noqa: F401
            n = n_samples or (10_000 if not use_gpu else 50_000)
            all_results["ngboost"] = bench_ngboost(n_samples=n, use_gpu=use_gpu)
        except ImportError:
            print("\n  ngboost not installed, skipping. Install: pip install ngboost")

    if "ebm" in benches:
        try:
            import interpret  # noqa: F401
            n = n_samples or (50_000 if use_gpu else 10_000)
            all_results["ebm"] = bench_ebm(n_samples=n, use_gpu=use_gpu)
        except ImportError:
            print("\n  interpret not installed, skipping. Install: pip install interpret")

    return all_results


# =============================================================================
# Modal entry points
# =============================================================================

if modal is not None and app is not None:

    @app.function(gpu="A100", image=image, timeout=3600)
    def _run_on_gpu(bench=None, n_samples=None):
        return run_all(use_gpu=True, bench=bench, n_samples=n_samples)

    @app.local_entrypoint()
    def main(bench: str = None, n_samples: int = None):
        """Run benchmarks on Modal A100."""
        if bench:
            print(f"Running '{bench}' benchmark on Modal A100...")
        else:
            print("Running all benchmarks on Modal A100...")

        results = _run_on_gpu.remote(bench=bench, n_samples=n_samples)

        # Save results
        results_dir = PROJECT_ROOT / "benchmarks" / "results"
        results_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_file = results_dir / f"gpu_benchmark_{timestamp}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_file}")


# =============================================================================
# Local execution
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="OpenBoost GPU Benchmarks")
    parser.add_argument("--local", action="store_true", help="Run locally on CPU")
    parser.add_argument(
        "--bench", choices=["xgboost", "ngboost", "ebm"],
        help="Run a single benchmark (default: all)",
    )
    parser.add_argument("--n-samples", type=int, help="Override dataset size")
    args = parser.parse_args()

    if args.local:
        sys.path.insert(0, str(PROJECT_ROOT / "src"))
        run_all(use_gpu=False, bench=args.bench, n_samples=args.n_samples)
    else:
        print("Usage:")
        print("  Modal:  uv run modal run benchmarks/compare_gpu.py")
        print("  Modal:  uv run modal run benchmarks/compare_gpu.py --bench xgboost")
        print("  Local:  uv run python benchmarks/compare_gpu.py --local")
        print("  Local:  uv run python benchmarks/compare_gpu.py --local --bench ngboost")
