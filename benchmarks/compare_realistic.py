"""Realistic large-scale benchmark: OpenBoost vs XGBoost.

Generates datasets that mimic real-world GBDT workloads: mixed feature
distributions, non-linear interactions, noise, redundant features, and
missing values. Runs at multiple scales to show how training time scales.

Usage:
    uv run python benchmarks/compare_realistic.py
    uv run python benchmarks/compare_realistic.py --quick
    uv run python benchmarks/compare_realistic.py --scale large
    uv run python benchmarks/compare_realistic.py --task regression
    uv run python benchmarks/compare_realistic.py --no-xgboost
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


# =============================================================================
# Dataset generators — mimic real tabular data
# =============================================================================

def make_realistic_regression(
    n_samples: int = 200_000,
    n_features: int = 50,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate regression data mimicking real tabular datasets.

    Features include:
    - Gaussian continuous (like age, income)
    - Uniform continuous (like percentages, ratios)
    - Log-normal (like prices, counts — right-skewed)
    - Binary indicators (like flags, boolean columns)
    - Ordinal (like ratings 1-5, education level)
    - Correlated feature pairs
    Target: non-linear function with interactions, thresholds, and noise.
    """
    rng = np.random.RandomState(seed)

    X = np.empty((n_samples, n_features), dtype=np.float32)

    # Assign feature types in blocks
    n_gauss = n_features // 5          # 20% gaussian
    n_uniform = n_features // 5        # 20% uniform
    n_lognorm = n_features // 10       # 10% log-normal
    n_binary = n_features // 10        # 10% binary
    n_ordinal = n_features // 10       # 10% ordinal
    n_corr = n_features // 10          # 10% correlated with earlier features
    n_noise = n_features - n_gauss - n_uniform - n_lognorm - n_binary - n_ordinal - n_corr  # rest is noise

    idx = 0
    # Gaussian features (standardized)
    X[:, idx:idx + n_gauss] = rng.randn(n_samples, n_gauss).astype(np.float32)
    gauss_end = idx + n_gauss
    idx = gauss_end

    # Uniform features [0, 1]
    X[:, idx:idx + n_uniform] = rng.uniform(0, 1, (n_samples, n_uniform)).astype(np.float32)
    idx += n_uniform

    # Log-normal features (right-skewed, like prices)
    X[:, idx:idx + n_lognorm] = rng.lognormal(0, 1, (n_samples, n_lognorm)).astype(np.float32)
    idx += n_lognorm

    # Binary indicator features
    X[:, idx:idx + n_binary] = rng.binomial(1, 0.3, (n_samples, n_binary)).astype(np.float32)
    binary_start = idx
    idx += n_binary

    # Ordinal features (1-5 scale)
    X[:, idx:idx + n_ordinal] = rng.randint(1, 6, (n_samples, n_ordinal)).astype(np.float32)
    idx += n_ordinal

    # Correlated features (linear combination of earlier features + noise)
    for j in range(n_corr):
        src = rng.randint(0, gauss_end)
        X[:, idx + j] = (0.8 * X[:, src] + 0.2 * rng.randn(n_samples)).astype(np.float32)
    idx += n_corr

    # Pure noise features
    X[:, idx:idx + n_noise] = rng.randn(n_samples, n_noise).astype(np.float32)

    # Target: non-linear function with interactions and thresholds
    y = (
        2.0 * np.sin(X[:, 0] * 1.5)                        # non-linear
        + 1.5 * X[:, 1] * X[:, 2]                           # interaction
        + np.where(X[:, 3] > 0, X[:, 4], -X[:, 4])          # threshold interaction
        + 0.8 * X[:, gauss_end] ** 2                         # quadratic on uniform
        + 0.5 * np.log1p(np.abs(X[:, gauss_end + n_uniform]))  # log transform on lognormal
        + 1.0 * X[:, binary_start]                           # binary effect
        + 0.3 * X[:, 0] * X[:, binary_start]                # continuous x binary interaction
        + rng.randn(n_samples).astype(np.float32) * 0.5     # noise
    ).astype(np.float32)

    # Inject missing values
    if missing_rate > 0:
        mask = rng.random((n_samples, n_features)) < missing_rate
        X[mask] = np.nan

    name = f"Realistic Regression ({n_samples:,} x {n_features})"
    return X, y, name


def make_realistic_binary(
    n_samples: int = 200_000,
    n_features: int = 50,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate binary classification data mimicking click/fraud/churn prediction.

    Imbalanced classes (~20% positive), non-linear decision boundary,
    feature interactions driving the signal.
    """
    rng = np.random.RandomState(seed)

    X, latent, _ = make_realistic_regression(n_samples, n_features, missing_rate=0.0, seed=seed)

    # Convert to binary via logistic function with class imbalance
    prob = 1.0 / (1.0 + np.exp(-(latent - np.percentile(latent, 70))))
    y = rng.binomial(1, prob).astype(np.float32)

    if missing_rate > 0:
        mask = rng.random((n_samples, n_features)) < missing_rate
        X[mask] = np.nan

    pos_rate = y.mean()
    name = f"Realistic Binary ({n_samples:,} x {n_features}, {pos_rate:.0%} pos)"
    return X, y, name


def make_realistic_multiclass(
    n_samples: int = 200_000,
    n_features: int = 50,
    n_classes: int = 5,
    missing_rate: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate multiclass data mimicking product category / segment prediction."""
    rng = np.random.RandomState(seed)

    X, _, _ = make_realistic_regression(n_samples, n_features, missing_rate=0.0, seed=seed)

    # Generate class logits from different feature subsets
    logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    features_per_class = max(3, n_features // n_classes)
    for c in range(n_classes):
        start = (c * features_per_class) % n_features
        end = min(start + features_per_class, n_features)
        weights = rng.randn(end - start).astype(np.float32) * 0.5
        logits[:, c] = X[:, start:end] @ weights
        # Add non-linearity
        logits[:, c] += 0.3 * np.sin(X[:, c % n_features] * 2)

    # Softmax + sample
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(n_classes, p=p) for p in probs], dtype=np.int32)

    if missing_rate > 0:
        mask = rng.random((n_samples, n_features)) < missing_rate
        X[mask] = np.nan

    name = f"Realistic {n_classes}-class ({n_samples:,} x {n_features})"
    return X, y, name


# =============================================================================
# Scale configurations
# =============================================================================

SCALES = {
    "small": {
        "n_samples": 50_000,
        "n_features": 30,
        "n_trees": 100,
        "max_depth": 6,
        "trials": 3,
    },
    "medium": {
        "n_samples": 200_000,
        "n_features": 50,
        "n_trees": 200,
        "max_depth": 6,
        "trials": 3,
    },
    "large": {
        "n_samples": 500_000,
        "n_features": 80,
        "n_trees": 300,
        "max_depth": 8,
        "trials": 2,
    },
    "xlarge": {
        "n_samples": 1_000_000,
        "n_features": 100,
        "n_trees": 500,
        "max_depth": 8,
        "trials": 1,
    },
}


# =============================================================================
# Benchmark runners
# =============================================================================

@dataclass
class TimingResult:
    trials: list[float] = field(default_factory=list)

    @property
    def median(self) -> float:
        if not self.trials:
            return float("inf")
        s = sorted(self.trials)
        return s[len(s) // 2]

    @property
    def mean(self) -> float:
        return sum(self.trials) / len(self.trials) if self.trials else float("inf")

    def fmt(self) -> str:
        if not self.trials:
            return "N/A"
        if len(self.trials) == 1:
            return f"{self.trials[0]:.3f}s"
        return f"{self.median:.3f}s (med of {len(self.trials)})"


@dataclass
class Result:
    model: str
    train: TimingResult
    predict: TimingResult
    metrics: dict[str, float]


def run_openboost(
    X_train, y_train, X_test, y_test,
    task: str, n_trees: int, max_depth: int, trials: int,
) -> Result:
    """Benchmark OpenBoost."""
    import openboost as ob

    # Warmup JIT
    n_warm = min(1000, len(X_train))
    if task == "multiclass":
        n_classes = len(np.unique(y_train))
        ob.MultiClassGradientBoosting(
            n_classes=n_classes, n_trees=2, max_depth=3,
        ).fit(X_train[:n_warm], y_train[:n_warm])
    elif task == "binary":
        ob.GradientBoosting(
            n_trees=2, max_depth=3, loss="logloss",
        ).fit(X_train[:n_warm], y_train[:n_warm])
    else:
        ob.GradientBoosting(
            n_trees=2, max_depth=3, loss="mse",
        ).fit(X_train[:n_warm], y_train[:n_warm])

    train_times, pred_times = [], []
    model = None

    for t in range(trials):
        if task == "multiclass":
            n_classes = len(np.unique(y_train))
            model = ob.MultiClassGradientBoosting(
                n_classes=n_classes, n_trees=n_trees,
                max_depth=max_depth, learning_rate=0.1,
            )
        elif task == "binary":
            model = ob.GradientBoosting(
                n_trees=n_trees, max_depth=max_depth,
                learning_rate=0.1, loss="logloss",
            )
        else:
            model = ob.GradientBoosting(
                n_trees=n_trees, max_depth=max_depth,
                learning_rate=0.1, loss="mse",
            )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        if task == "multiclass":
            pred = np.argmax(model.predict_proba(X_test), axis=1)
        elif task == "binary":
            raw = model.predict(X_test)
            pred = (1 / (1 + np.exp(-raw)) > 0.5).astype(int)
        else:
            pred = model.predict(X_test)
        pred_times.append(time.perf_counter() - t0)

        print(f"    trial {t + 1}/{trials}: {train_times[-1]:.3f}s")

    metrics = _compute_metrics(y_test, pred, task)
    return Result("OpenBoost", TimingResult(train_times), TimingResult(pred_times), metrics)


def run_xgboost(
    X_train, y_train, X_test, y_test,
    task: str, n_trees: int, max_depth: int, trials: int,
) -> Result | None:
    """Benchmark XGBoost. Returns None if not installed."""
    try:
        import xgboost as xgb
    except ImportError:
        return None

    # Warmup
    n_warm = min(1000, len(X_train))
    if task in ("binary", "multiclass"):
        xgb.XGBClassifier(
            n_estimators=2, max_depth=3, tree_method="hist", verbosity=0,
        ).fit(X_train[:n_warm], y_train[:n_warm])
    else:
        xgb.XGBRegressor(
            n_estimators=2, max_depth=3, tree_method="hist", verbosity=0,
        ).fit(X_train[:n_warm], y_train[:n_warm])

    train_times, pred_times = [], []
    pred = None

    for t in range(trials):
        if task == "multiclass":
            model = xgb.XGBClassifier(
                n_estimators=n_trees, max_depth=max_depth,
                learning_rate=0.1, tree_method="hist", verbosity=0,
            )
        elif task == "binary":
            model = xgb.XGBClassifier(
                n_estimators=n_trees, max_depth=max_depth,
                learning_rate=0.1, tree_method="hist", verbosity=0,
                objective="binary:logistic",
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_trees, max_depth=max_depth,
                learning_rate=0.1, tree_method="hist", verbosity=0,
            )

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_times.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        pred = model.predict(X_test)
        pred_times.append(time.perf_counter() - t0)

        print(f"    trial {t + 1}/{trials}: {train_times[-1]:.3f}s")

    metrics = _compute_metrics(y_test, pred, task)
    return Result("XGBoost", TimingResult(train_times), TimingResult(pred_times), metrics)


def _compute_metrics(y_true, y_pred, task: str) -> dict[str, float]:
    if task == "regression":
        mse = float(np.mean((y_true - y_pred) ** 2))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        return {"RMSE": float(np.sqrt(mse)), "R2": round(r2, 4)}
    else:
        acc = float(np.mean(y_true == y_pred))
        return {"Accuracy": round(acc, 4)}


# =============================================================================
# Output
# =============================================================================

def print_comparison(
    dataset_name: str,
    scale_name: str,
    ob: Result,
    xgb: Result | None,
    cfg: dict,
):
    """Print formatted comparison."""
    print()
    print(f"  {dataset_name}")
    print(f"  Scale: {scale_name} | {cfg['n_samples']:,} samples, "
          f"{cfg['n_features']} features, {cfg['n_trees']} trees, depth={cfg['max_depth']}")
    print(f"  {'─' * 70}")
    print(f"  {'Model':<14} {'Train (median)':<22} {'Predict':<18} ", end="")
    for k in ob.metrics:
        print(f"{k:<12}", end="")
    print()
    print(f"  {'─' * 70}")

    # OpenBoost row
    print(f"  {'OpenBoost':<14} {ob.train.fmt():<22} {ob.predict.fmt():<18} ", end="")
    for v in ob.metrics.values():
        print(f"{v:<12.4f}", end="")
    print()

    # XGBoost row
    if xgb:
        print(f"  {'XGBoost':<14} {xgb.train.fmt():<22} {xgb.predict.fmt():<18} ", end="")
        for v in xgb.metrics.values():
            print(f"{v:<12.4f}", end="")
        print()

        speedup = xgb.train.median / ob.train.median
        faster = "OpenBoost" if speedup > 1 else "XGBoost"
        ratio = max(speedup, 1 / speedup)
        print(f"  {'─' * 70}")
        print(f"  Speedup: {ratio:.2f}x ({faster} faster)")

    print()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Realistic large-scale OpenBoost vs XGBoost benchmark")
    parser.add_argument("--scale", choices=list(SCALES.keys()) + ["all", "sweep"],
                        default="medium", help="Dataset scale (default: medium)")
    parser.add_argument("--task", choices=["regression", "binary", "multiclass", "all"],
                        default="regression", help="Task type (default: regression)")
    parser.add_argument("--quick", action="store_true", help="Quick mode: small scale, 1 trial")
    parser.add_argument("--no-xgboost", action="store_true", help="Skip XGBoost")
    parser.add_argument("--missing", type=float, default=0.0,
                        help="Fraction of missing values (default: 0, try 0.05)")
    args = parser.parse_args()

    if args.quick:
        args.scale = "small"

    tasks = ["regression", "binary", "multiclass"] if args.task == "all" else [args.task]
    scales = list(SCALES.keys()) if args.scale in ("all", "sweep") else [args.scale]

    print("=" * 74)
    print("  Realistic Benchmark: OpenBoost vs XGBoost")
    print("=" * 74)

    import openboost as ob
    print(f"  Backend:    {ob.get_backend()}")
    has_xgb = not args.no_xgboost
    if has_xgb:
        try:
            import xgboost
            print(f"  XGBoost:    {xgboost.__version__}")
        except ImportError:
            print("  XGBoost:    not installed (skipping)")
            has_xgb = False
    print(f"  Missing:    {args.missing:.0%}" if args.missing > 0 else "  Missing:    none")
    print()

    summary_rows = []

    for task in tasks:
        for scale_name in scales:
            cfg = SCALES[scale_name].copy()
            if args.quick:
                cfg["trials"] = 1
                cfg["n_trees"] = min(cfg["n_trees"], 50)

            print(f"{'=' * 74}")
            print(f"  Task: {task} | Scale: {scale_name}")
            print(f"{'=' * 74}")

            # Generate data
            print(f"  Generating {cfg['n_samples']:,} x {cfg['n_features']} dataset...")
            if task == "regression":
                X, y, name = make_realistic_regression(
                    cfg["n_samples"], cfg["n_features"], args.missing)
            elif task == "binary":
                X, y, name = make_realistic_binary(
                    cfg["n_samples"], cfg["n_features"], args.missing)
            else:
                X, y, name = make_realistic_multiclass(
                    cfg["n_samples"], cfg["n_features"], missing_rate=args.missing)

            # Split
            split = int(len(X) * 0.8)
            rng = np.random.RandomState(42)
            idx = rng.permutation(len(X))
            X_train, X_test = X[idx[:split]], X[idx[split:]]
            y_train, y_test = y[idx[:split]], y[idx[split:]]

            print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}")
            print(f"  Test:  {X_test.shape[0]:,} x {X_test.shape[1]}")

            # OpenBoost
            print(f"\n  OpenBoost ({cfg['trials']} trial{'s' if cfg['trials'] > 1 else ''}):")
            ob_result = run_openboost(
                X_train, y_train, X_test, y_test,
                task, cfg["n_trees"], cfg["max_depth"], cfg["trials"],
            )

            # XGBoost
            xgb_result = None
            if has_xgb:
                print(f"\n  XGBoost ({cfg['trials']} trial{'s' if cfg['trials'] > 1 else ''}):")
                xgb_result = run_xgboost(
                    X_train, y_train, X_test, y_test,
                    task, cfg["n_trees"], cfg["max_depth"], cfg["trials"],
                )

            print_comparison(name, scale_name, ob_result, xgb_result, cfg)

            # Collect for summary
            speedup = (xgb_result.train.median / ob_result.train.median) if xgb_result else None
            summary_rows.append({
                "task": task, "scale": scale_name,
                "samples": cfg["n_samples"], "features": cfg["n_features"],
                "trees": cfg["n_trees"],
                "ob_time": ob_result.train.median,
                "xgb_time": xgb_result.train.median if xgb_result else None,
                "speedup": speedup,
            })

    # Final summary
    print("=" * 74)
    print("  SUMMARY")
    print("=" * 74)
    print(f"  {'Task':<14} {'Scale':<8} {'Samples':>10} {'OB':>10} {'XGB':>10} {'Speedup':>10}")
    print(f"  {'─' * 68}")
    for r in summary_rows:
        xgb_str = f"{r['xgb_time']:.3f}s" if r["xgb_time"] else "N/A"
        if r["speedup"]:
            faster = "OB" if r["speedup"] > 1 else "XGB"
            spd_str = f"{max(r['speedup'], 1/r['speedup']):.2f}x {faster}"
        else:
            spd_str = "N/A"
        print(f"  {r['task']:<14} {r['scale']:<8} {r['samples']:>10,} "
              f"{r['ob_time']:>9.3f}s {xgb_str:>10} {spd_str:>10}")
    print()


if __name__ == "__main__":
    main()
