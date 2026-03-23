"""Performance regression check for CI.

Runs a fixed, small benchmark and compares against stored baselines.
Fails if any metric degrades by more than 20%.

Usage:
    uv run python benchmarks/check_performance.py
    uv run python benchmarks/check_performance.py --update-baselines
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASELINE_FILE = Path(__file__).parent / "results" / "performance_baselines.json"

# Regression threshold: fail if metric exceeds baseline by this factor
REGRESSION_THRESHOLD = 1.20  # 20%


def _generate_data(n_samples=5000, n_features=10, seed=42):
    """Generate fixed synthetic dataset."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
         + rng.randn(n_samples).astype(np.float32) * 0.1).astype(np.float32)
    return X, y


def run_fixed_benchmark():
    """Run fixed benchmark and return results dict."""
    import openboost as ob

    X, y = _generate_data()
    n_trees = 100
    max_depth = 6

    # Measure fit time (median of 3 trials)
    fit_times = []
    for _ in range(3):
        model = ob.GradientBoosting(
            n_trees=n_trees, max_depth=max_depth, learning_rate=0.1
        )
        t0 = time.perf_counter()
        model.fit(X, y)
        fit_times.append(time.perf_counter() - t0)

    # Measure predict time
    predict_times = []
    for _ in range(3):
        t0 = time.perf_counter()
        model.predict(X)
        predict_times.append(time.perf_counter() - t0)

    # Measure peak memory
    tracemalloc.start()
    model2 = ob.GradientBoosting(
        n_trees=n_trees, max_depth=max_depth, learning_rate=0.1
    )
    model2.fit(X, y)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    # Measure accuracy
    pred = model.predict(X)
    mse = float(np.mean((pred - y) ** 2))
    r2 = float(1 - np.sum((pred - y) ** 2) / np.sum((y - np.mean(y)) ** 2))

    return {
        "fit_time_median": float(sorted(fit_times)[1]),
        "predict_time_median": float(sorted(predict_times)[1]),
        "peak_memory_mb": float(peak_mb),
        "mse": mse,
        "r2": r2,
        "n_samples": len(X),
        "n_features": X.shape[1],
        "n_trees": n_trees,
        "max_depth": max_depth,
    }


def save_baselines(results):
    """Save results as new baselines."""
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(BASELINE_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Baselines saved to {BASELINE_FILE}")


def load_baselines():
    """Load stored baselines."""
    with open(BASELINE_FILE) as f:
        return json.load(f)


def check_regression(results, baselines):
    """Compare results against baselines. Returns list of regressions."""
    regressions = []

    # Time metrics: fail if current > baseline * threshold
    for metric in ["fit_time_median", "predict_time_median"]:
        if results[metric] > baselines[metric] * REGRESSION_THRESHOLD:
            regressions.append(
                f"  {metric}: {results[metric]:.4f}s > "
                f"{baselines[metric]:.4f}s * {REGRESSION_THRESHOLD} = "
                f"{baselines[metric] * REGRESSION_THRESHOLD:.4f}s"
            )

    # Memory: fail if current > baseline * threshold
    if results["peak_memory_mb"] > baselines["peak_memory_mb"] * REGRESSION_THRESHOLD:
        regressions.append(
            f"  peak_memory_mb: {results['peak_memory_mb']:.2f}MB > "
            f"{baselines['peak_memory_mb']:.2f}MB * {REGRESSION_THRESHOLD}"
        )

    # Accuracy: fail if MSE increases (model got worse)
    if results["mse"] > baselines["mse"] * REGRESSION_THRESHOLD:
        regressions.append(
            f"  mse: {results['mse']:.6f} > "
            f"{baselines['mse']:.6f} * {REGRESSION_THRESHOLD}"
        )

    return regressions


def main():
    parser = argparse.ArgumentParser(description="Performance regression check")
    parser.add_argument(
        "--update-baselines", action="store_true",
        help="Update baselines with current results"
    )
    args = parser.parse_args()

    print("Running fixed benchmark...")
    results = run_fixed_benchmark()

    print(f"  fit_time:     {results['fit_time_median']:.4f}s")
    print(f"  predict_time: {results['predict_time_median']:.4f}s")
    print(f"  peak_memory:  {results['peak_memory_mb']:.2f}MB")
    print(f"  mse:          {results['mse']:.6f}")
    print(f"  r2:           {results['r2']:.4f}")

    if args.update_baselines:
        save_baselines(results)
        return

    if not BASELINE_FILE.exists():
        print(f"\nNo baselines found at {BASELINE_FILE}")
        print("Run with --update-baselines to create them.")
        save_baselines(results)
        return

    baselines = load_baselines()
    regressions = check_regression(results, baselines)

    if regressions:
        print(f"\nPerformance regression detected ({REGRESSION_THRESHOLD:.0%} threshold):")
        for r in regressions:
            print(r)
        sys.exit(1)
    else:
        print("\nNo performance regressions detected.")


if __name__ == "__main__":
    main()
