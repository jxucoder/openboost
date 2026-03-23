"""OpenBoost Benchmark Suite.

A single, self-contained benchmark that works locally with zero external
dependencies beyond the project's [bench] extras.

Usage:
    uv run python benchmarks/compare_cpu.py
    uv run python benchmarks/compare_cpu.py --quick
    uv run python benchmarks/compare_cpu.py --task regression
    uv run python benchmarks/compare_cpu.py --trials 5
    uv run python benchmarks/compare_cpu.py --n-samples 100000

Options:
    --task          Run a specific task: regression, binary, multiclass, all (default: all)
    --trials        Number of timing trials per benchmark (default: 3)
    --n-samples     Override dataset size (default: auto based on device)
    --quick         Quick mode: fewer trees, smaller data, 1 trial
    --no-xgboost    Skip XGBoost comparison
    --verbose       Print per-trial timings
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Result Types
# =============================================================================


@dataclass
class TimingResult:
    """Holds multiple trial timings and computes statistics."""

    trials: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return mean(self.trials) if self.trials else float("inf")

    @property
    def std(self) -> float:
        return stdev(self.trials) if len(self.trials) >= 2 else 0.0

    def fmt(self, unit: str = "s") -> str:
        if not self.trials:
            return "N/A"
        if unit == "ms":
            return f"{self.mean * 1000:.1f} ± {self.std * 1000:.1f}ms"
        return f"{self.mean:.3f} ± {self.std:.3f}s"


@dataclass
class BenchmarkResult:
    model_name: str
    train_time: TimingResult
    pred_time: TimingResult
    metrics: dict[str, float]


# =============================================================================
# Datasets — real data bundled with sklearn, no downloads needed
# =============================================================================


def load_regression_data(n_samples: int | None = None):
    """California Housing — 20640 samples, 8 features."""
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split

    data = fetch_california_housing()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    if n_samples and n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, "California Housing"


def load_binary_data(n_samples: int | None = None):
    """Breast Cancer — 569 samples, 30 features. If n_samples > 569, augment."""
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split

    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    if n_samples and n_samples > len(X):
        rng = np.random.RandomState(42)
        repeats = (n_samples // len(X)) + 1
        X = np.tile(X, (repeats, 1))[:n_samples]
        y = np.tile(y, repeats)[:n_samples]
        noise = rng.randn(*X.shape).astype(np.float32) * 0.01
        X = X + noise
    elif n_samples and n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, "Breast Cancer"


def load_multiclass_data(n_samples: int | None = None):
    """Digits — 1797 samples, 64 features, 10 classes. Augment if needed."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    data = load_digits()
    X, y = data.data.astype(np.float32), data.target.astype(np.int32)

    if n_samples and n_samples > len(X):
        rng = np.random.RandomState(42)
        repeats = (n_samples // len(X)) + 1
        X = np.tile(X, (repeats, 1))[:n_samples]
        y = np.tile(y, repeats)[:n_samples]
        noise = rng.randn(*X.shape).astype(np.float32) * 0.1
        X = X + noise
    elif n_samples and n_samples < len(X):
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, "Digits (10-class)"


def load_synthetic_regression(n_samples: int = 50_000, n_features: int = 20):
    """Synthetic non-linear regression for scaling tests."""
    from sklearn.model_selection import train_test_split

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (
        np.sin(X[:, 0] * 2)
        + 0.5 * X[:, 1] ** 2
        + 0.3 * X[:, 2] * X[:, 3]
        + 0.1 * rng.randn(n_samples)
    ).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, f"Synthetic ({n_samples:,} x {n_features})"


# =============================================================================
# Benchmark runners
# =============================================================================


def _warmup_openboost(X_train, y_train, task: str, n_trees: int = 3, max_depth: int = 3):
    """Warmup JIT compilation for OpenBoost."""
    import openboost as ob

    n_warm = min(500, len(X_train))
    X_w, y_w = X_train[:n_warm], y_train[:n_warm]

    if task == "multiclass":
        n_classes = len(np.unique(y_train))
        ob.MultiClassGradientBoosting(
            n_classes=n_classes, n_trees=n_trees, max_depth=max_depth
        ).fit(X_w, y_w)
    elif task == "binary":
        ob.GradientBoosting(
            n_trees=n_trees, max_depth=max_depth, loss="logloss"
        ).fit(X_w, y_w)
    else:
        ob.GradientBoosting(
            n_trees=n_trees, max_depth=max_depth, loss="mse"
        ).fit(X_w, y_w)


def _warmup_xgboost(X_train, y_train, task: str, n_trees: int = 3, max_depth: int = 3):
    """Warmup XGBoost to be fair."""
    import xgboost as xgb

    n_warm = min(500, len(X_train))
    X_w, y_w = X_train[:n_warm], y_train[:n_warm]

    if task in ("binary", "multiclass"):
        xgb.XGBClassifier(
            n_estimators=n_trees, max_depth=max_depth, tree_method="hist", verbosity=0
        ).fit(X_w, y_w)
    else:
        xgb.XGBRegressor(
            n_estimators=n_trees, max_depth=max_depth, tree_method="hist", verbosity=0
        ).fit(X_w, y_w)


def benchmark_openboost(
    X_train, X_test, y_train, y_test,
    task: str,
    n_trees: int = 100,
    max_depth: int = 6,
    n_trials: int = 3,
    verbose: bool = False,
) -> BenchmarkResult:
    """Benchmark OpenBoost with multiple trials."""
    import openboost as ob
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

    _warmup_openboost(X_train, y_train, task)

    train_times: list[float] = []
    pred_times: list[float] = []
    last_y_pred = None
    last_y_pred_proba = None

    for trial in range(n_trials):
        if task == "multiclass":
            n_classes = len(np.unique(y_train))
            model = ob.MultiClassGradientBoosting(
                n_classes=n_classes,
                n_trees=n_trees,
                max_depth=max_depth,
                learning_rate=0.1,
            )
        elif task == "binary":
            model = ob.GradientBoosting(
                n_trees=n_trees,
                max_depth=max_depth,
                learning_rate=0.1,
                loss="logloss",
            )
        else:
            model = ob.GradientBoosting(
                n_trees=n_trees,
                max_depth=max_depth,
                learning_rate=0.1,
                loss="mse",
            )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        if task == "multiclass":
            last_y_pred_proba = model.predict_proba(X_test)
            last_y_pred = np.argmax(last_y_pred_proba, axis=1)
        elif task == "binary":
            raw = model.predict(X_test)
            last_y_pred_proba = 1 / (1 + np.exp(-raw))
            last_y_pred = (last_y_pred_proba > 0.5).astype(int)
        else:
            last_y_pred = model.predict(X_test)
        pred_times.append(time.perf_counter() - start)

        if verbose:
            print(f"    trial {trial + 1}: train={train_times[-1]:.3f}s  pred={pred_times[-1] * 1000:.1f}ms")

    metrics: dict[str, float] = {}
    if task == "regression":
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_test, last_y_pred)))
        metrics["R²"] = float(r2_score(y_test, last_y_pred))
    elif task == "binary":
        metrics["Accuracy"] = float(accuracy_score(y_test, last_y_pred))
        try:
            metrics["AUC"] = float(roc_auc_score(y_test, last_y_pred_proba))
        except ValueError:
            pass
    else:
        metrics["Accuracy"] = float(accuracy_score(y_test, last_y_pred))

    return BenchmarkResult(
        model_name="OpenBoost",
        train_time=TimingResult(train_times),
        pred_time=TimingResult(pred_times),
        metrics=metrics,
    )


def benchmark_xgboost(
    X_train, X_test, y_train, y_test,
    task: str,
    n_trees: int = 100,
    max_depth: int = 6,
    n_trials: int = 3,
    verbose: bool = False,
) -> BenchmarkResult | None:
    """Benchmark XGBoost with multiple trials. Returns None if not installed."""
    try:
        import xgboost as xgb
    except ImportError:
        return None

    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, roc_auc_score

    _warmup_xgboost(X_train, y_train, task)

    train_times: list[float] = []
    pred_times: list[float] = []
    last_y_pred = None
    last_y_pred_proba = None

    for trial in range(n_trials):
        if task == "multiclass":
            n_classes = len(np.unique(y_train))
            model = xgb.XGBClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                learning_rate=0.1,
                tree_method="hist",
                verbosity=0,
                objective="multi:softprob",
                num_class=n_classes,
            )
        elif task == "binary":
            model = xgb.XGBClassifier(
                n_estimators=n_trees,
                max_depth=max_depth,
                learning_rate=0.1,
                tree_method="hist",
                verbosity=0,
                objective="binary:logistic",
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_trees,
                max_depth=max_depth,
                learning_rate=0.1,
                tree_method="hist",
                verbosity=0,
            )

        start = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        if task in ("binary", "multiclass"):
            last_y_pred_proba = model.predict_proba(X_test)
            last_y_pred = model.predict(X_test)
            if task == "binary":
                last_y_pred_proba = last_y_pred_proba[:, 1]
        else:
            last_y_pred = model.predict(X_test)
        pred_times.append(time.perf_counter() - start)

        if verbose:
            print(f"    trial {trial + 1}: train={train_times[-1]:.3f}s  pred={pred_times[-1] * 1000:.1f}ms")

    metrics: dict[str, float] = {}
    if task == "regression":
        metrics["RMSE"] = float(np.sqrt(mean_squared_error(y_test, last_y_pred)))
        metrics["R²"] = float(r2_score(y_test, last_y_pred))
    elif task == "binary":
        metrics["Accuracy"] = float(accuracy_score(y_test, last_y_pred))
        try:
            metrics["AUC"] = float(roc_auc_score(y_test, last_y_pred_proba))
        except ValueError:
            pass
    else:
        metrics["Accuracy"] = float(accuracy_score(y_test, last_y_pred))

    return BenchmarkResult(
        model_name="XGBoost",
        train_time=TimingResult(train_times),
        pred_time=TimingResult(pred_times),
        metrics=metrics,
    )


# =============================================================================
# Output formatting
# =============================================================================


def print_header(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def print_comparison(dataset_name: str, ob_result: BenchmarkResult, xgb_result: BenchmarkResult | None) -> None:
    """Print a nicely formatted comparison table."""
    print(f"\n  Dataset: {dataset_name}")
    print(f"  {'─' * 64}")

    header = f"  {'Model':<14} {'Train':<22} {'Predict':<22}"
    for key in ob_result.metrics:
        header += f" {key:<10}"
    print(header)
    print(f"  {'─' * 64}")

    # OpenBoost row
    row = f"  {'OpenBoost':<14} {ob_result.train_time.fmt():<22} {ob_result.pred_time.fmt('ms'):<22}"
    for val in ob_result.metrics.values():
        row += f" {val:<10.4f}"
    print(row)

    # XGBoost row
    if xgb_result:
        row = f"  {'XGBoost':<14} {xgb_result.train_time.fmt():<22} {xgb_result.pred_time.fmt('ms'):<22}"
        for val in xgb_result.metrics.values():
            row += f" {val:<10.4f}"
        print(row)

        # Speedup
        speedup = xgb_result.train_time.mean / ob_result.train_time.mean
        faster = "OpenBoost" if speedup > 1 else "XGBoost"
        print(f"  {'─' * 64}")
        print(f"  Train speedup: {speedup:.2f}x ({faster} faster)")

    print()


# =============================================================================
# Main
# =============================================================================


def run_benchmarks(
    tasks: list[str],
    n_trials: int = 3,
    n_trees: int = 100,
    max_depth: int = 6,
    n_samples: int | None = None,
    skip_xgboost: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full benchmark suite.

    Args:
        tasks: List of task types to benchmark.
        n_trials: Number of timing trials per benchmark.
        n_trees: Number of boosting rounds.
        max_depth: Maximum tree depth.
        n_samples: Override dataset size (None = use natural size).
        skip_xgboost: Whether to skip XGBoost comparison.
        verbose: Print per-trial timings.

    Returns:
        Dictionary of results keyed by task name.
    """
    import openboost as ob

    print_header("OpenBoost Benchmark Suite")
    print(f"  Backend:    {ob.get_backend()}")
    print(f"  Trials:     {n_trials}")
    print(f"  Trees:      {n_trees}")
    print(f"  Max depth:  {max_depth}")
    if n_samples:
        print(f"  N samples:  {n_samples:,}")

    has_xgb = not skip_xgboost
    if has_xgb:
        try:
            import xgboost
            print(f"  XGBoost:    {xgboost.__version__}")
        except ImportError:
            print("  XGBoost:    not installed (skipping)")
            has_xgb = False

    all_results: dict[str, Any] = {}

    loaders = {
        "regression": (load_regression_data, "regression"),
        "binary": (load_binary_data, "binary"),
        "multiclass": (load_multiclass_data, "multiclass"),
    }

    for task_name in tasks:
        if task_name not in loaders:
            print(f"\n  Unknown task: {task_name}, skipping")
            continue

        loader, task_type = loaders[task_name]
        print_header(f"Task: {task_name}")

        X_train, X_test, y_train, y_test, dataset_name = loader(n_samples)

        print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
        print(f"  Test:  {X_test.shape[0]:,} x {X_test.shape[1]}")

        # OpenBoost
        print(f"\n  Running OpenBoost ({n_trials} trials)...")
        ob_result = benchmark_openboost(
            X_train, X_test, y_train, y_test,
            task=task_type,
            n_trees=n_trees,
            max_depth=max_depth,
            n_trials=n_trials,
            verbose=verbose,
        )

        # XGBoost
        xgb_result = None
        if has_xgb:
            print(f"  Running XGBoost ({n_trials} trials)...")
            xgb_result = benchmark_xgboost(
                X_train, X_test, y_train, y_test,
                task=task_type,
                n_trees=n_trees,
                max_depth=max_depth,
                n_trials=n_trials,
                verbose=verbose,
            )

        print_comparison(dataset_name, ob_result, xgb_result)
        all_results[task_name] = {
            "dataset": dataset_name,
            "openboost": ob_result,
            "xgboost": xgb_result,
        }

    # Also run synthetic regression if regression is in tasks and n_samples wasn't tiny
    if "regression" in tasks:
        synth_size = n_samples or 50_000
        if synth_size >= 10_000:
            print_header("Task: regression (synthetic, scaling test)")
            X_train, X_test, y_train, y_test, dataset_name = load_synthetic_regression(synth_size)
            print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
            print(f"  Test:  {X_test.shape[0]:,} x {X_test.shape[1]}")

            print(f"\n  Running OpenBoost ({n_trials} trials)...")
            ob_result = benchmark_openboost(
                X_train, X_test, y_train, y_test,
                task="regression",
                n_trees=n_trees,
                max_depth=max_depth,
                n_trials=n_trials,
                verbose=verbose,
            )
            xgb_result = None
            if has_xgb:
                print(f"  Running XGBoost ({n_trials} trials)...")
                xgb_result = benchmark_xgboost(
                    X_train, X_test, y_train, y_test,
                    task="regression",
                    n_trees=n_trees,
                    max_depth=max_depth,
                    n_trials=n_trials,
                    verbose=verbose,
                )
            print_comparison(dataset_name, ob_result, xgb_result)
            all_results["regression_synthetic"] = {
                "dataset": dataset_name,
                "openboost": ob_result,
                "xgboost": xgb_result,
            }

    # Final summary
    print_header("Summary")
    print(f"  {'Task':<28} {'OB Train':<18} {'XGB Train':<18} {'Speedup':<12}")
    print(f"  {'─' * 74}")
    for task_name, res in all_results.items():
        ob_t = res["openboost"].train_time.fmt()
        xgb_t = res["xgboost"].train_time.fmt() if res["xgboost"] else "N/A"
        if res["xgboost"]:
            spd = res["xgboost"].train_time.mean / res["openboost"].train_time.mean
            faster = "OB" if spd > 1 else "XGB"
            spd_str = f"{spd:.2f}x ({faster})"
        else:
            spd_str = "N/A"
        print(f"  {task_name:<28} {ob_t:<18} {xgb_t:<18} {spd_str:<12}")

    print()
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="OpenBoost Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--task",
        choices=["regression", "binary", "multiclass", "all"],
        default="all",
        help="Which task to benchmark (default: all)",
    )
    parser.add_argument(
        "--trials", type=int, default=3,
        help="Number of timing trials (default: 3)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=None,
        help="Override dataset size",
    )
    parser.add_argument(
        "--n-trees", type=int, default=100,
        help="Number of boosting rounds (default: 100)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=6,
        help="Max tree depth (default: 6)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick mode: 1 trial, 20 trees, small data",
    )
    parser.add_argument(
        "--no-xgboost", action="store_true",
        help="Skip XGBoost comparison",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-trial timings",
    )

    args = parser.parse_args()

    if args.task == "all":
        tasks = ["regression", "binary", "multiclass"]
    else:
        tasks = [args.task]

    n_trials = args.trials
    n_trees = args.n_trees
    n_samples = args.n_samples

    if args.quick:
        n_trials = 1
        n_trees = 20
        if n_samples is None:
            n_samples = 5_000

    run_benchmarks(
        tasks=tasks,
        n_trials=n_trials,
        n_trees=n_trees,
        max_depth=args.max_depth,
        n_samples=n_samples,
        skip_xgboost=args.no_xgboost,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
