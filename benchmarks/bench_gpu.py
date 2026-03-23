"""Fair GPU benchmark: OpenBoost vs XGBoost.

Principles for fairness:
  1. Both receive raw numpy arrays (no pre-built DMatrix)
  2. cuda.synchronize() after OpenBoost fit() to ensure all kernels complete
  3. Both at default threading (XGBoost uses all CPU cores — that's its design)
  4. JIT/CUDA context warmup for both before timing
  5. Time only model.fit(), not data generation or prediction
  6. Same hyperparameters: n_trees, max_depth, learning_rate, max_bin=256
  7. Multiple trials, report median

Runs locally if CUDA is available, or on Modal A100.

Usage:
    # Local (requires CUDA GPU)
    uv run python benchmarks/bench_gpu.py
    uv run python benchmarks/bench_gpu.py --scale large
    uv run python benchmarks/bench_gpu.py --scale sweep
    uv run python benchmarks/bench_gpu.py --task all

    # Modal A100
    uv run modal run benchmarks/bench_gpu.py
    uv run modal run benchmarks/bench_gpu.py --scale large
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Realistic data generators (same as compare_realistic.py)
# =============================================================================

def _make_features(n_samples: int, n_features: int, rng) -> np.ndarray:
    """Mixed feature types: gaussian, uniform, lognormal, binary, ordinal, correlated."""
    X = np.empty((n_samples, n_features), dtype=np.float32)
    n_gauss = n_features // 5
    n_uniform = n_features // 5
    n_lognorm = n_features // 10
    n_binary = n_features // 10
    n_ordinal = n_features // 10
    n_corr = n_features // 10
    n_noise = n_features - n_gauss - n_uniform - n_lognorm - n_binary - n_ordinal - n_corr

    idx = 0
    X[:, idx:idx + n_gauss] = rng.randn(n_samples, n_gauss)
    gauss_end = idx + n_gauss
    idx = gauss_end
    X[:, idx:idx + n_uniform] = rng.uniform(0, 1, (n_samples, n_uniform))
    idx += n_uniform
    X[:, idx:idx + n_lognorm] = rng.lognormal(0, 1, (n_samples, n_lognorm))
    idx += n_lognorm
    X[:, idx:idx + n_binary] = rng.binomial(1, 0.3, (n_samples, n_binary))
    binary_start = idx
    idx += n_binary
    X[:, idx:idx + n_ordinal] = rng.randint(1, 6, (n_samples, n_ordinal))
    idx += n_ordinal
    for j in range(n_corr):
        src = rng.randint(0, gauss_end)
        X[:, idx + j] = 0.8 * X[:, src] + 0.2 * rng.randn(n_samples)
    idx += n_corr
    X[:, idx:idx + n_noise] = rng.randn(n_samples, n_noise)
    return X.astype(np.float32), gauss_end, binary_start


def make_regression(n_samples: int, n_features: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    X, gauss_end, binary_start = _make_features(n_samples, n_features, rng)
    y = (
        2.0 * np.sin(X[:, 0] * 1.5)
        + 1.5 * X[:, 1] * X[:, 2]
        + np.where(X[:, 3] > 0, X[:, 4], -X[:, 4])
        + 0.8 * X[:, gauss_end] ** 2
        + 1.0 * X[:, binary_start]
        + rng.randn(n_samples).astype(np.float32) * 0.5
    ).astype(np.float32)
    return X, y


def make_binary(n_samples: int, n_features: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    X, y_latent = make_regression(n_samples, n_features, seed)[:2]
    prob = 1.0 / (1.0 + np.exp(-(y_latent - np.percentile(y_latent, 70))))
    y = rng.binomial(1, prob).astype(np.float32)
    return X, y


# =============================================================================
# Scale configs
# =============================================================================

SCALES = {
    "small":  {"n_samples": 500_000,    "n_features": 50,  "n_trees": 200,  "max_depth": 6,  "trials": 3},
    "medium": {"n_samples": 2_000_000,  "n_features": 80,  "n_trees": 300,  "max_depth": 8,  "trials": 3},
    "large":  {"n_samples": 5_000_000,  "n_features": 100, "n_trees": 500,  "max_depth": 8,  "trials": 2},
    "xlarge": {"n_samples": 10_000_000, "n_features": 100, "n_trees": 500,  "max_depth": 10, "trials": 1},
}


# =============================================================================
# Result type
# =============================================================================

@dataclass
class TimingResult:
    times: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        s = sorted(self.times)
        return s[len(s) // 2] if s else float("inf")

    def fmt(self) -> str:
        if not self.times:
            return "N/A"
        if len(self.times) == 1:
            return f"{self.times[0]:.3f}s"
        return f"{self.median:.3f}s (med/{len(self.times)})"


# =============================================================================
# Benchmark runners
# =============================================================================

def warmup_openboost(X, y, task: str, max_depth: int):
    """Warmup Numba JIT + CUDA context."""
    import openboost as ob
    from numba import cuda

    n = min(1000, len(X))
    for _ in range(2):  # Two rounds to ensure all kernels compiled
        if task == "binary":
            ob.GradientBoosting(n_trees=3, max_depth=max_depth, loss="logloss").fit(X[:n], y[:n])
        else:
            ob.GradientBoosting(n_trees=3, max_depth=max_depth, loss="mse").fit(X[:n], y[:n])
    cuda.synchronize()


def warmup_xgboost(X, y, task: str, max_depth: int):
    """Warmup XGBoost GPU."""
    import xgboost as xgb

    n = min(1000, len(X))
    if task == "binary":
        xgb.XGBClassifier(
            n_estimators=3, max_depth=max_depth,
            tree_method="hist", device="cuda", verbosity=0,
        ).fit(X[:n], y[:n])
    else:
        xgb.XGBRegressor(
            n_estimators=3, max_depth=max_depth,
            tree_method="hist", device="cuda", verbosity=0,
        ).fit(X[:n], y[:n])


def bench_openboost(
    X_train, y_train, task: str, n_trees: int, max_depth: int, trials: int,
) -> TimingResult:
    """Time OpenBoost fit() with GPU sync."""
    import openboost as ob
    from numba import cuda

    times = []
    for t in range(trials):
        if task == "binary":
            model = ob.GradientBoosting(
                n_trees=n_trees, max_depth=max_depth,
                learning_rate=0.1, loss="logloss",
            )
        else:
            model = ob.GradientBoosting(
                n_trees=n_trees, max_depth=max_depth,
                learning_rate=0.1, loss="mse",
            )

        cuda.synchronize()  # Ensure GPU idle before timing
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        cuda.synchronize()  # Ensure all GPU work complete
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    OB  trial {t + 1}/{trials}: {elapsed:.3f}s")

    return TimingResult(times)


def bench_xgboost(
    X_train, y_train, task: str, n_trees: int, max_depth: int, trials: int,
) -> TimingResult:
    """Time XGBoost fit() on GPU. Default threading (all cores)."""
    import xgboost as xgb

    times = []
    for t in range(trials):
        if task == "binary":
            model = xgb.XGBClassifier(
                n_estimators=n_trees, max_depth=max_depth,
                learning_rate=0.1, tree_method="hist", device="cuda",
                verbosity=0, max_bin=256,
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_trees, max_depth=max_depth,
                learning_rate=0.1, tree_method="hist", device="cuda",
                verbosity=0, max_bin=256,
            )

        # Sync GPU before timing for consistency with OpenBoost measurement
        try:
            from numba import cuda as nb_cuda
            nb_cuda.synchronize()
        except Exception:
            pass

        t0 = time.perf_counter()
        model.fit(X_train, y_train)

        # XGBoost should sync internally, but force a device sync to be sure
        try:
            from numba import cuda as nb_cuda
            nb_cuda.synchronize()
        except Exception:
            pass

        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        print(f"    XGB trial {t + 1}/{trials}: {elapsed:.3f}s")

    return TimingResult(times)


def compute_accuracy(model, X_test, y_test, task: str, lib: str) -> dict[str, float]:
    """Quick accuracy check to ensure both models are actually learning."""
    if task == "regression":
        if lib == "xgboost":
            pred = model.predict(X_test)
        else:
            pred = model.predict(X_test)
        mse = float(np.mean((y_test - pred) ** 2))
        ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
        ss_res = float(np.sum((y_test - pred) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        return {"RMSE": round(float(np.sqrt(mse)), 4), "R2": round(r2, 4)}
    else:
        if lib == "xgboost":
            pred = model.predict(X_test)
        else:
            raw = model.predict(X_test)
            pred = (1 / (1 + np.exp(-raw)) > 0.5).astype(int)
        acc = float(np.mean(y_test == pred))
        return {"Accuracy": round(acc, 4)}


# =============================================================================
# Main benchmark
# =============================================================================

def run_benchmark(
    task: str, scale_name: str, cfg: dict, skip_xgb: bool = False,
) -> dict:
    """Run one task at one scale. Returns results dict."""
    import openboost as ob
    from numba import cuda

    n_samples = cfg["n_samples"]
    n_features = cfg["n_features"]
    n_trees = cfg["n_trees"]
    max_depth = cfg["max_depth"]
    trials = cfg["trials"]

    print(f"\n{'=' * 74}")
    print(f"  {task.upper()} | {scale_name} | {n_samples:,} x {n_features} | "
          f"{n_trees} trees | depth={max_depth} | {trials} trial(s)")
    print(f"  Backend: {ob.get_backend()} | GPU: ", end="")
    try:
        gpu_name = cuda.get_current_device().name
        print(gpu_name.decode() if isinstance(gpu_name, bytes) else gpu_name)
    except Exception:
        print("unknown")
    print(f"{'=' * 74}")

    # Generate data
    print(f"  Generating dataset...")
    if task == "binary":
        X, y = make_binary(n_samples, n_features)
    else:
        X, y = make_regression(n_samples, n_features)

    # Split 80/20
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(X))
    split = int(len(X) * 0.8)
    X_train, X_test = X[idx[:split]], X[idx[split:]]
    y_train, y_test = y[idx[:split]], y[idx[split:]]
    print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
    print(f"  Test:  {X_test.shape[0]:,} x {X_test.shape[1]}")

    # Warmup OpenBoost
    print(f"\n  Warming up OpenBoost (JIT + CUDA context)...")
    warmup_openboost(X_train, y_train, task, max_depth)

    # Warmup XGBoost
    has_xgb = not skip_xgb
    if has_xgb:
        try:
            print(f"  Warming up XGBoost (GPU context)...")
            warmup_xgboost(X_train, y_train, task, max_depth)
        except Exception as e:
            print(f"  XGBoost GPU warmup failed: {e}")
            has_xgb = False

    # Benchmark OpenBoost
    print(f"\n  OpenBoost ({trials} trial{'s' if trials > 1 else ''}):")
    ob_timing = bench_openboost(X_train, y_train, task, n_trees, max_depth, trials)

    # Train one final model for accuracy check
    import openboost as ob
    if task == "binary":
        ob_model = ob.GradientBoosting(n_trees=n_trees, max_depth=max_depth, learning_rate=0.1, loss="logloss")
    else:
        ob_model = ob.GradientBoosting(n_trees=n_trees, max_depth=max_depth, learning_rate=0.1, loss="mse")
    ob_model.fit(X_train, y_train)
    ob_metrics = compute_accuracy(ob_model, X_test, y_test, task, "openboost")

    # Benchmark XGBoost
    xgb_timing = None
    xgb_metrics = {}
    if has_xgb:
        print(f"\n  XGBoost ({trials} trial{'s' if trials > 1 else ''}):")
        xgb_timing = bench_xgboost(X_train, y_train, task, n_trees, max_depth, trials)

        import xgboost as xgb
        if task == "binary":
            xgb_model = xgb.XGBClassifier(
                n_estimators=n_trees, max_depth=max_depth, learning_rate=0.1,
                tree_method="hist", device="cuda", verbosity=0, max_bin=256,
            )
        else:
            xgb_model = xgb.XGBRegressor(
                n_estimators=n_trees, max_depth=max_depth, learning_rate=0.1,
                tree_method="hist", device="cuda", verbosity=0, max_bin=256,
            )
        xgb_model.fit(X_train, y_train)
        xgb_metrics = compute_accuracy(xgb_model, X_test, y_test, task, "xgboost")

    # Print comparison
    print(f"\n  {'─' * 70}")
    print(f"  {'Model':<14} {'Fit time':<22}", end="")
    for k in ob_metrics:
        print(f" {k:<12}", end="")
    print()
    print(f"  {'─' * 70}")

    print(f"  {'OpenBoost':<14} {ob_timing.fmt():<22}", end="")
    for v in ob_metrics.values():
        print(f" {v:<12}", end="")
    print()

    if xgb_timing:
        print(f"  {'XGBoost':<14} {xgb_timing.fmt():<22}", end="")
        for v in xgb_metrics.values():
            print(f" {v:<12}", end="")
        print()

        speedup = xgb_timing.median / ob_timing.median
        faster = "OpenBoost" if speedup > 1 else "XGBoost"
        ratio = max(speedup, 1.0 / speedup)
        print(f"  {'─' * 70}")
        print(f"  Result: {ratio:.2f}x ({faster} faster)")

    result = {
        "task": task, "scale": scale_name,
        "n_samples": n_samples, "n_features": n_features,
        "n_trees": n_trees, "max_depth": max_depth,
        "ob_median_s": ob_timing.median,
        "ob_times": ob_timing.times,
        "ob_metrics": ob_metrics,
    }
    if xgb_timing:
        result["xgb_median_s"] = xgb_timing.median
        result["xgb_times"] = xgb_timing.times
        result["xgb_metrics"] = xgb_metrics
        result["speedup"] = speedup

    return result


def run_all(tasks: list[str], scales: list[str], skip_xgb: bool = False) -> list[dict]:
    """Run benchmarks across tasks and scales."""
    import openboost as ob

    print("=" * 74)
    print("  Fair GPU Benchmark: OpenBoost vs XGBoost")
    print("=" * 74)
    print(f"  OpenBoost backend: {ob.get_backend()}")

    if ob.get_backend() != "cuda":
        print("\n  WARNING: Not running on CUDA backend!")
        print("  Set OPENBOOST_BACKEND=cuda or ensure a GPU is available.")
        print("  Results on CPU are not comparable.\n")

    if not skip_xgb:
        try:
            import xgboost
            print(f"  XGBoost version:   {xgboost.__version__}")
        except ImportError:
            print("  XGBoost: not installed, skipping")
            skip_xgb = True

    print(f"  Tasks:  {', '.join(tasks)}")
    print(f"  Scales: {', '.join(scales)}")
    print()
    print("  Fairness controls:")
    print("    - Both receive raw numpy (no pre-built DMatrix)")
    print("    - cuda.synchronize() after OpenBoost fit()")
    print("    - Both at default threading (best-case for each)")
    print("    - XGBoost max_bin=256 (matching OpenBoost)")
    print("    - JIT/GPU warmup before timing")

    results = []
    for task in tasks:
        for scale_name in scales:
            cfg = SCALES[scale_name]
            result = run_benchmark(task, scale_name, cfg, skip_xgb=skip_xgb)
            results.append(result)

    # Summary table
    print(f"\n{'=' * 74}")
    print("  SUMMARY")
    print(f"{'=' * 74}")
    print(f"  {'Task':<12} {'Scale':<8} {'Samples':>10} {'Trees':>6} "
          f"{'OB':>10} {'XGB':>10} {'Speedup':>12}")
    print(f"  {'─' * 72}")
    for r in results:
        ob_str = f"{r['ob_median_s']:.3f}s"
        xgb_str = f"{r.get('xgb_median_s', 0):.3f}s" if "xgb_median_s" in r else "N/A"
        if "speedup" in r:
            sp = r["speedup"]
            faster = "OB" if sp > 1 else "XGB"
            spd_str = f"{max(sp, 1/sp):.2f}x {faster}"
        else:
            spd_str = "N/A"
        print(f"  {r['task']:<12} {r['scale']:<8} {r['n_samples']:>10,} {r['n_trees']:>6} "
              f"{ob_str:>10} {xgb_str:>10} {spd_str:>12}")
    print()

    return results


# =============================================================================
# Modal support
# =============================================================================

try:
    import modal

    modal_app = modal.App("openboost-bench-gpu")

    modal_image = (
        modal.Image.from_registry(
            "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12"
        )
        .pip_install(
            "numpy>=1.24",
            "numba>=0.60",
            "joblib>=1.3",
            "xgboost>=2.0",
            "scikit-learn>=1.3",
        )
        .add_local_dir(
            str(PROJECT_ROOT / "src" / "openboost"),
            remote_path="/root/openboost_pkg/openboost",
        )
    )

    @modal_app.function(gpu="A100", image=modal_image, timeout=3600)
    def _run_on_modal(tasks, scales, skip_xgb):
        sys.path.insert(0, "/root/openboost_pkg")
        import openboost as ob
        ob.set_backend("cuda")
        return run_all(tasks, scales, skip_xgb)

    @modal_app.local_entrypoint()
    def modal_main(
        task: str = "regression",
        scale: str = "medium",
        no_xgboost: bool = False,
    ):
        tasks = ["regression", "binary"] if task == "all" else [task]
        scales = list(SCALES.keys()) if scale == "sweep" else [scale]
        print(f"Running on Modal A100: tasks={tasks}, scales={scales}")
        results = _run_on_modal.remote(tasks, scales, no_xgboost)

        results_dir = PROJECT_ROOT / "benchmarks" / "results"
        results_dir.mkdir(exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = results_dir / f"gpu_fair_{ts}.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_file}")

except ImportError:
    modal = None


# =============================================================================
# Local execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Fair GPU benchmark: OpenBoost vs XGBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--task", choices=["regression", "binary", "all"],
                        default="regression")
    parser.add_argument("--scale", choices=list(SCALES.keys()) + ["sweep"],
                        default="medium")
    parser.add_argument("--no-xgboost", action="store_true")
    args = parser.parse_args()

    tasks = ["regression", "binary"] if args.task == "all" else [args.task]
    scales = list(SCALES.keys()) if args.scale == "sweep" else [args.scale]

    results = run_all(tasks, scales, skip_xgb=args.no_xgboost)

    # Save results
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = results_dir / f"gpu_fair_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    main()
