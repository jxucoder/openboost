"""Performance regression check for CI or a local baseline.

Runs a fixed, small benchmark and compares against stored baselines.
Fails if any metric degrades by more than 20%.

Usage:
    uv run python benchmarks/check_performance.py --baseline baseline.json
    uv run python benchmarks/check_performance.py --benchmark-only --output result.json
    uv run python benchmarks/check_performance.py --update-baselines
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

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


def collect_provenance(source_root: Path) -> dict[str, str]:
    """Collect enough environment data to interpret a raw CI result."""
    import openboost as ob

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source_root,
        check=False,
        capture_output=True,
        text=True,
    )
    git_commit = commit.stdout.strip() if commit.returncode == 0 else "unknown"

    return {
        "git_commit": git_commit,
        "openboost_version": ob.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
        "openboost_backend": os.environ.get("OPENBOOST_BACKEND", "auto"),
        "numba_num_threads": os.environ.get("NUMBA_NUM_THREADS", "default"),
        "numba_cache_dir": os.environ.get("NUMBA_CACHE_DIR", "default"),
    }


def save_results(results, path: Path):
    """Save benchmark results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {path}")


def load_baselines(path: Path):
    """Load stored baselines."""
    with open(path) as f:
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
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_FILE,
        help="Baseline JSON to compare against",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write the current raw benchmark result to this JSON file",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Run and save the benchmark without comparing it",
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root whose src/openboost implementation should run",
    )
    args = parser.parse_args()

    if (
        not args.update_baselines
        and not args.benchmark_only
        and not args.baseline.exists()
    ):
        print(f"No baseline found at {args.baseline}", file=sys.stderr)
        print(
            "Pass --baseline, use --benchmark-only, or explicitly create a "
            "local baseline with --update-baselines.",
            file=sys.stderr,
        )
        sys.exit(2)

    source_dir = args.source_root.resolve() / "src"
    if not source_dir.is_dir():
        parser.error(f"source root has no src directory: {args.source_root}")
    sys.path.insert(0, str(source_dir))

    print("Running fixed benchmark...")
    results = run_fixed_benchmark()
    results["benchmark_schema_version"] = 1
    results["provenance"] = collect_provenance(args.source_root.resolve())

    print(f"  fit_time:     {results['fit_time_median']:.4f}s")
    print(f"  predict_time: {results['predict_time_median']:.4f}s")
    print(f"  peak_memory:  {results['peak_memory_mb']:.2f}MB")
    print(f"  mse:          {results['mse']:.6f}")
    print(f"  r2:           {results['r2']:.4f}")

    if args.output:
        save_results(results, args.output)

    if args.benchmark_only:
        return

    if args.update_baselines:
        save_results(results, args.baseline)
        return

    baselines = load_baselines(args.baseline)
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
