"""Scaling analysis for OpenBoost.

Measures how training time scales with n_samples and n_features.
Computes scaling exponents to verify sub-quadratic behavior.

Usage:
    uv run python benchmarks/check_scaling.py
    uv run python benchmarks/check_scaling.py --quick
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import openboost as ob  # noqa: E402


def run_scaling_analysis(quick=False):
    """Run scaling analysis across n_samples and n_features."""

    if quick:
        sample_grid = [1_000, 5_000, 10_000]
        feature_grid = [10, 50]
        n_trees = 20
    else:
        sample_grid = [1_000, 5_000, 10_000, 50_000, 100_000]
        feature_grid = [10, 50, 100]
        n_trees = 50

    max_depth = 6
    learning_rate = 0.1

    results = []

    print(f"{'n_samples':>10} {'n_features':>10} {'fit_time':>10} {'pred_time':>10}")
    print("-" * 45)

    for n_features in feature_grid:
        for n_samples in sample_grid:
            rng = np.random.RandomState(42)
            X = rng.randn(n_samples, n_features).astype(np.float32)
            y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n_samples).astype(np.float32) * 0.1).astype(np.float32)

            # Fit timing (single run for large data, 3 runs for small)
            trials = 3 if n_samples <= 10_000 else 1
            fit_times = []
            for _ in range(trials):
                m = ob.GradientBoosting(
                    n_trees=n_trees, max_depth=max_depth, learning_rate=learning_rate
                )
                t0 = time.perf_counter()
                m.fit(X, y)
                fit_times.append(time.perf_counter() - t0)

            # Predict timing
            pred_times = []
            for _ in range(3):
                t0 = time.perf_counter()
                m.predict(X)
                pred_times.append(time.perf_counter() - t0)

            fit_time = sorted(fit_times)[len(fit_times) // 2]
            pred_time = sorted(pred_times)[1]

            results.append({
                "n_samples": n_samples,
                "n_features": n_features,
                "fit_time": fit_time,
                "pred_time": pred_time,
            })

            print(f"{n_samples:>10} {n_features:>10} {fit_time:>10.4f}s {pred_time:>10.4f}s")

    # Compute scaling exponents
    print("\n" + "=" * 50)
    print("Scaling Exponents (log(time) = alpha * log(n_samples) + beta)")
    print("  alpha ≈ 1.0: linear scaling (optimal)")
    print("  alpha ≈ 1.5: O(n^1.5) (acceptable)")
    print("  alpha ≈ 2.0: quadratic (bad)")
    print("=" * 50)

    for n_features in feature_grid:
        subset = [r for r in results if r["n_features"] == n_features]
        if len(subset) < 3:
            continue

        log_n = np.log([r["n_samples"] for r in subset])
        log_t = np.log([r["fit_time"] for r in subset])

        # Linear regression: log_t = alpha * log_n + beta
        alpha, beta = np.polyfit(log_n, log_t, 1)

        print(f"\n  n_features={n_features}: alpha = {alpha:.2f}")
        if alpha < 1.3:
            print("    -> Near-linear scaling")
        elif alpha < 1.7:
            print("    -> Slightly super-linear")
        else:
            print("    -> WARNING: scaling appears quadratic or worse")


def main():
    parser = argparse.ArgumentParser(description="OpenBoost Scaling Analysis")
    parser.add_argument("--quick", action="store_true", help="Quick mode with fewer data points")
    args = parser.parse_args()

    run_scaling_analysis(quick=args.quick)


if __name__ == "__main__":
    main()
