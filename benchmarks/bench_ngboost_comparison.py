"""Benchmark: OpenBoost NaturalBoost vs NGBoost (Normal distribution).

Head-to-head comparison of ``openboost.NaturalBoostNormal`` against
``ngboost.NGBRegressor`` with a Normal distribution, on:

(a) Synthetic heteroscedastic regression (Friedman #1 signal with
    feature-dependent noise: y = f(x) + sigma(x) * eps) at 10K and 50K samples.
(b) California Housing (sklearn ``fetch_california_housing``; skipped with a
    note if the dataset cannot be downloaded/loaded offline).

Both models get the same fixed seed (42), identical train/test splits, and a
comparable budget: 500 boosting rounds, learning rate 0.03, depth-3 trees.
Metrics: test Gaussian NLL (same closed-form formula for both), CRPS
(``openboost.crps_gaussian`` applied to both models' predicted Normal params),
point RMSE, and wall-clock fit time.

The comparison is run CPU-vs-CPU (NGBoost is CPU-only). OpenBoost additionally
has a GPU tree path that this benchmark does NOT measure.

Usage:
    OPENBOOST_BACKEND=cpu uv run --with ngboost python benchmarks/bench_ngboost_comparison.py
    OPENBOOST_BACKEND=cpu uv run --with ngboost python benchmarks/bench_ngboost_comparison.py --quick

Output: benchmarks/results/ngboost_comparison_<YYYYMMDD>.json
(rerunning on the same day overwrites the same file, so the run is idempotent).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Force an honest CPU-vs-CPU comparison (NGBoost is CPU-only). Must be set
# before openboost is imported.
os.environ.setdefault("OPENBOOST_BACKEND", "cpu")

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import openboost as ob

SEED = 42
RESULTS_DIR = Path(__file__).parent / "results"

# Shared training budget — deliberately close to NGBoost's defaults
# (n_estimators=500, depth-3 base learner) so neither library is tuned
# harder than the other.
N_TREES = 500
LEARNING_RATE = 0.03
MAX_DEPTH = 3

CONFIG_NOTES = (
    "Both models: Normal distribution, 500 boosting rounds, learning_rate=0.03, "
    "depth-3 trees, natural gradient, fixed seed 42, identical train/test splits. "
    "Remaining parameters are library defaults and differ structurally: OpenBoost "
    "fits histogram-based trees (n_bins=254, min_child_weight=1.0, reg_lambda=1.0); "
    "NGBoost fits exact-split sklearn DecisionTreeRegressor(criterion='friedman_mse') "
    "base learners with a per-round line search (its default learner). Timings are "
    "single-run wall clock on CPU; OpenBoost's one-time Numba JIT compilation is "
    "excluded via a small untimed warmup fit (both libraries get the same warmup)."
)


# =============================================================================
# Datasets
# =============================================================================

def make_heteroscedastic(n_samples: int, n_features: int = 10, seed: int = SEED):
    """Synthetic heteroscedastic regression: y = f(x) + sigma(x) * eps.

    Signal is the Friedman #1 function on the first 5 of ``n_features``
    uniform features; noise scale grows linearly with feature 5, so the
    target's variance is feature-dependent (what distributional models
    are supposed to capture).
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0, 1, (n_samples, n_features))
    f = (
        10 * np.sin(np.pi * X[:, 0] * X[:, 1])
        + 20 * (X[:, 2] - 0.5) ** 2
        + 10 * X[:, 3]
        + 5 * X[:, 4]
    )
    sigma = 1.0 + 2.0 * X[:, 5]  # heteroscedastic: sd in [1, 3]
    y = f + sigma * rng.randn(n_samples)
    return X.astype(np.float64), y.astype(np.float64)


def load_datasets(quick: bool = False) -> tuple[list[dict], list[str]]:
    """Build the benchmark datasets. Returns (datasets, notes)."""
    notes: list[str] = []
    sizes = [2_000] if quick else [10_000, 50_000]
    datasets = [
        {
            "name": f"synthetic_heteroscedastic_{n:_}",
            "description": (
                "y = friedman1(x) + (1 + 2*x5) * eps, 10 uniform features, "
                f"n={n}, seed={SEED}"
            ),
            "data": make_heteroscedastic(n),
        }
        for n in sizes
    ]

    try:
        from sklearn.datasets import fetch_california_housing

        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        if quick:
            X, y = X[:2_000], y[:2_000]
        datasets.append(
            {
                "name": "california_housing",
                "description": (
                    f"sklearn fetch_california_housing, n={len(y)}, "
                    f"{X.shape[1]} features"
                ),
                "data": (X.astype(np.float64), y.astype(np.float64)),
            }
        )
    except Exception as exc:  # network/cache failure — skip, don't crash
        msg = f"California Housing skipped (fetch failed: {type(exc).__name__}: {exc})"
        print(f"WARNING: {msg}")
        notes.append(msg)

    return datasets, notes


# =============================================================================
# Models
# =============================================================================

def make_openboost():
    """OpenBoost NaturalBoost with Normal distribution at the shared budget.

    n_bins=254 is the effective library default (the default 256 gets clamped
    to 254 with a warning because bin 255 is reserved for missing values);
    passing it explicitly is behaviorally identical and keeps output clean.
    """
    return ob.NaturalBoostNormal(
        n_trees=N_TREES,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        n_bins=254,
    )


def make_ngboost():
    """NGBoost NGBRegressor with Normal distribution at the shared budget.

    Base learner is NGBoost's own default (depth-3 friedman_mse CART) made
    explicit so the config is fully documented.
    """
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor

    return NGBRegressor(
        Dist=Normal,
        n_estimators=N_TREES,
        learning_rate=LEARNING_RATE,
        Base=DecisionTreeRegressor(criterion="friedman_mse", max_depth=MAX_DEPTH),
        natural_gradient=True,
        verbose=False,
        random_state=SEED,
    )


def predict_normal_params(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, std) of the predicted Normal for either library."""
    if hasattr(model, "predict_distribution"):  # OpenBoost
        params = model.predict_distribution(X).params
    else:  # NGBoost
        params = model.pred_dist(X).params
    return np.asarray(params["loc"], dtype=np.float64), np.asarray(
        params["scale"], dtype=np.float64
    )


# =============================================================================
# Metrics — identical formulas for both libraries
# =============================================================================

def gaussian_nll(y: np.ndarray, mean: np.ndarray, std: np.ndarray) -> float:
    """Mean Gaussian negative log-likelihood (shared formula for fairness)."""
    std = np.clip(std, 1e-12, None)
    return float(
        np.mean(0.5 * np.log(2 * np.pi * std**2) + (y - mean) ** 2 / (2 * std**2))
    )


def evaluate(model, name: str, X_tr, y_tr, X_te, y_te) -> dict:
    """Fit + score one model; returns metrics dict."""
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0

    mean, std = predict_normal_params(model, X_te)
    result = {
        "fit_time_s": round(fit_time, 3),
        "test_nll": round(gaussian_nll(y_te, mean, std), 4),
        "test_crps": round(ob.crps_gaussian(y_te, mean, std), 4),
        "test_rmse": round(float(np.sqrt(np.mean((y_te - mean) ** 2))), 4),
    }
    print(
        f"    {name:<12} fit {result['fit_time_s']:>8.2f}s   "
        f"NLL {result['test_nll']:>8.4f}   CRPS {result['test_crps']:>7.4f}   "
        f"RMSE {result['test_rmse']:>7.4f}"
    )
    return result


# =============================================================================
# Runner
# =============================================================================

def warmup() -> None:
    """Untimed warmup so OpenBoost's one-time Numba JIT compile (and any
    first-call overhead in either library) is excluded from timed runs."""
    X, y = make_heteroscedastic(512, seed=0)
    make_openboost().fit(X[:256], y[:256])
    small_ngb = make_ngboost()
    small_ngb.set_params(n_estimators=5)
    small_ngb.fit(X[:256], y[:256])


def run(quick: bool = False) -> Path:
    import ngboost
    import scipy
    import sklearn
    from sklearn.model_selection import train_test_split

    datasets, notes = load_datasets(quick=quick)

    print("Warming up (untimed JIT compile)...")
    warmup()

    results = []
    for ds in datasets:
        X, y = ds["data"]
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=SEED
        )
        print(f"\n{ds['name']}  (train={len(y_tr)}, test={len(y_te)}, "
              f"features={X.shape[1]})")

        ob_metrics = evaluate(make_openboost(), "openboost", X_tr, y_tr, X_te, y_te)
        ngb_metrics = evaluate(make_ngboost(), "ngboost", X_tr, y_tr, X_te, y_te)

        results.append(
            {
                "dataset": ds["name"],
                "description": ds["description"],
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "n_features": int(X.shape[1]),
                "openboost": ob_metrics,
                "ngboost": ngb_metrics,
                "speedup_openboost_vs_ngboost": round(
                    ngb_metrics["fit_time_s"] / ob_metrics["fit_time_s"], 2
                ),
                "nll_delta_openboost_minus_ngboost": round(
                    ob_metrics["test_nll"] - ngb_metrics["test_nll"], 4
                ),
            }
        )

    report = {
        "benchmark": "naturalboost_vs_ngboost",
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": SEED,
        "quick_mode": quick,
        "backend": ob.get_backend(),
        "backend_note": (
            "CPU-vs-CPU comparison (NGBoost is CPU-only). OpenBoost also has a "
            "GPU tree path (fit_tree_gpu_native) that this benchmark does NOT "
            "measure."
        ),
        "platform": {
            "python": platform.python_version(),
            "system": f"{platform.system()} {platform.machine()}",
            "processor": platform.processor(),
            "cpu_count": os.cpu_count(),
        },
        "versions": {
            "openboost": ob.__version__,
            "ngboost": ngboost.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
        "configs": {
            "shared": {
                "n_boosting_rounds": N_TREES,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "distribution": "Normal",
                "train_test_split": {"test_size": 0.2, "random_state": SEED},
            },
            "openboost": {
                "model": "NaturalBoostNormal",
                "n_trees": N_TREES,
                "learning_rate": LEARNING_RATE,
                "max_depth": MAX_DEPTH,
                "min_child_weight": 1.0,
                "reg_lambda": 1.0,
                "n_bins": 254,
            },
            "ngboost": {
                "model": "NGBRegressor",
                "Dist": "Normal",
                "n_estimators": N_TREES,
                "learning_rate": LEARNING_RATE,
                "Base": "DecisionTreeRegressor(criterion='friedman_mse', max_depth=3)",
                "natural_gradient": True,
                "random_state": SEED,
            },
            "notes": CONFIG_NOTES,
        },
        "metrics_note": (
            "test_nll uses one shared closed-form Gaussian NLL for both models; "
            "test_crps uses openboost.crps_gaussian on both models' predicted "
            "(mean, std) — same formula, fair. test_rmse uses the predicted mean "
            "as the point estimate. Lower is better for all three."
        ),
        "skipped": notes,
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"ngboost_comparison_{datetime.now():%Y%m%d}.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenBoost NaturalBoost vs NGBoost benchmark"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Small smoke-test run (2K samples) to verify the script end-to-end",
    )
    args = parser.parse_args()
    run(quick=args.quick)


if __name__ == "__main__":
    main()
