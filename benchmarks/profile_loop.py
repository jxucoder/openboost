"""Profile OpenBoost training and identify bottlenecks.

Part of the self-recursive improvement loop:
  1. Run this script to profile and identify the top bottleneck
  2. Optimize the target code
  3. Re-run to verify improvement and compare with previous run

Usage:
    uv run python benchmarks/profile_loop.py
    uv run python benchmarks/profile_loop.py --n-samples 200000 --n-features 50
    uv run python benchmarks/profile_loop.py --n-trees 200 --max-depth 8
    uv run python benchmarks/profile_loop.py --summarize
    uv run python benchmarks/profile_loop.py --growth leafwise
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Profile OpenBoost training loop")
    parser.add_argument("--n-samples", type=int, default=50_000)
    parser.add_argument("--n-features", type=int, default=20)
    parser.add_argument("--n-trees", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--loss", type=str, default="mse")
    parser.add_argument("--output-dir", type=str, default="logs/")
    parser.add_argument("--summarize", action="store_true",
                        help="Print machine-readable summary for improvement loops")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Generate synthetic dataset
    rng = np.random.RandomState(args.seed)
    X = rng.randn(args.n_samples, args.n_features).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
         + rng.randn(args.n_samples).astype(np.float32) * 0.1).astype(np.float32)

    import openboost as ob
    from openboost._profiler import ProfilingCallback, print_profile_summary

    profiler = ProfilingCallback(output_dir=args.output_dir)
    model = ob.GradientBoosting(
        n_trees=args.n_trees,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        loss=args.loss,
    )

    print(f"Profiling: {args.n_samples:,} samples, {args.n_features} features, "
          f"{args.n_trees} trees, depth={args.max_depth}")

    # Warmup JIT (first fit compiles Numba kernels)
    warmup_model = ob.GradientBoosting(n_trees=2, max_depth=args.max_depth, loss=args.loss)
    warmup_model.fit(X[:1000], y[:1000])

    wall_start = time.perf_counter()
    model.fit(X, y, callbacks=[profiler])
    wall_time = time.perf_counter() - wall_start

    print(f"Wall time: {wall_time:.2f}s")
    print(f"Report: {profiler.report_path}")

    if args.summarize:
        print()
        report = profiler.report
        report["_path"] = str(profiler.report_path)
        print_profile_summary(report)
    else:
        # Print compact phase table
        report = profiler.report
        print(f"\n{'Phase':<20} {'Time (s)':>10} {'%':>8} {'Calls':>8}")
        print("-" * 50)
        for phase, data in report["phases"].items():
            calls = str(data["calls"]) if data["calls"] is not None else "-"
            print(f"{phase:<20} {data['total_s']:>10.3f} {data['pct']:>7.1f}% {calls:>8}")
        print("-" * 50)
        print(f"{'TOTAL':<20} {report['total_time_s']:>10.3f}")

        if report.get("bottlenecks"):
            print(f"\nTop bottleneck: {report['bottlenecks'][0]['phase']} "
                  f"({report['bottlenecks'][0]['pct']}%)")
            print(f"  Target: {report['bottlenecks'][0]['target']}")
            print(f"  Recommendation: {report['bottlenecks'][0]['recommendation']}")

        if report.get("comparison"):
            comp = report["comparison"]
            delta = comp["delta_total_pct"]
            sign = "+" if delta > 0 else ""
            print(f"\nvs previous run: {sign}{delta}% total time")


if __name__ == "__main__":
    main()
