"""Speed benchmarks for autoresearch v2.

Measures wall-clock training time across multiple scales and tasks,
comparing OpenBoost vs XGBoost on GPU.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .data_generators import SPEED_CONFIGS, make_speed_data
from .scoring import compute_speed_score

# Tree hyperparams per config
TREE_PARAMS = {
    "small_reg":  {"n_trees": 200, "max_depth": 6},
    "small_bin":  {"n_trees": 200, "max_depth": 6},
    "medium_reg": {"n_trees": 200, "max_depth": 8},
    "medium_bin": {"n_trees": 200, "max_depth": 8},
    "large_reg":  {"n_trees": 300, "max_depth": 8},
}

QUICK_CONFIGS = ["medium_reg"]


def _median(values: list[float]) -> float:
    s = sorted(values)
    return s[len(s) // 2]


def _warmup_openboost(X: np.ndarray, y: np.ndarray, task: str, max_depth: int):
    """Warmup JIT + CUDA with small data."""
    import openboost as ob
    from numba import cuda

    n = min(2000, len(X))
    loss = "logloss" if task == "binary" else "mse"
    for _ in range(2):
        ob.GradientBoosting(n_trees=3, max_depth=max_depth, loss=loss).fit(
            X[:n], y[:n]
        )
    cuda.synchronize()


def _warmup_xgboost(X: np.ndarray, y: np.ndarray, task: str, max_depth: int):
    """Warmup XGBoost CUDA with small data."""
    import xgboost as xgb

    n = min(2000, len(X))
    if task == "binary":
        model = xgb.XGBClassifier(
            n_estimators=3, max_depth=max_depth, tree_method="hist",
            device="cuda", max_bin=256, verbosity=0,
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=3, max_depth=max_depth, tree_method="hist",
            device="cuda", max_bin=256, verbosity=0,
        )
    model.fit(X[:n], y[:n])


def _time_openboost(
    X: np.ndarray, y: np.ndarray, task: str,
    n_trees: int, max_depth: int, trials: int,
) -> list[float]:
    """Time OpenBoost training, return list of elapsed times."""
    import openboost as ob
    from numba import cuda

    loss = "logloss" if task == "binary" else "mse"
    times = []
    for _ in range(trials):
        model = ob.GradientBoosting(
            n_trees=n_trees, max_depth=max_depth,
            learning_rate=0.1, loss=loss,
        )
        cuda.synchronize()
        t0 = time.perf_counter()
        model.fit(X, y)
        cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def _time_xgboost(
    X: np.ndarray, y: np.ndarray, task: str,
    n_trees: int, max_depth: int, trials: int,
) -> list[float]:
    """Time XGBoost training, return list of elapsed times."""
    import xgboost as xgb
    from numba import cuda

    times = []
    for _ in range(trials):
        if task == "binary":
            model = xgb.XGBClassifier(
                n_estimators=n_trees, max_depth=max_depth,
                learning_rate=0.1, tree_method="hist",
                device="cuda", max_bin=256, verbosity=0,
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_trees, max_depth=max_depth,
                learning_rate=0.1, tree_method="hist",
                device="cuda", max_bin=256, verbosity=0,
            )
        cuda.synchronize()
        t0 = time.perf_counter()
        model.fit(X, y)
        cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return times


def run_speed_benchmarks(
    trials: int = 3,
    include_xgb: bool = True,
    quick: bool = False,
) -> dict[str, Any]:
    """Run speed benchmarks across all configs.

    Returns dict with:
        score: float in [0, 1]
        gates_passed: bool
        configs: dict[config_name, {ob_median_s, xgb_median_s, ob_times, xgb_times, speedup}]
        geo_mean_speedup: float
    """
    configs = QUICK_CONFIGS if quick else list(SPEED_CONFIGS.keys())
    results: dict[str, dict] = {}
    warmed_up_ob = set()
    warmed_up_xgb = set()

    for config_name in configs:
        print(f"\n  [speed] {config_name}...")
        cfg = SPEED_CONFIGS[config_name]
        tree = TREE_PARAMS[config_name]
        task = cfg["task"]

        X, y, task = make_speed_data(config_name)

        # Warmup (once per task type)
        warmup_key = (task, tree["max_depth"])
        if warmup_key not in warmed_up_ob:
            _warmup_openboost(X, y, task, tree["max_depth"])
            warmed_up_ob.add(warmup_key)

        # Time OpenBoost
        ob_times = _time_openboost(X, y, task, tree["n_trees"], tree["max_depth"], trials)
        ob_median = _median(ob_times)
        print(f"    OB: {ob_median:.3f}s (times: {[f'{t:.3f}' for t in ob_times]})")

        entry = {
            "ob_median_s": ob_median,
            "ob_times": ob_times,
            "xgb_median_s": None,
            "xgb_times": None,
            "speedup": None,
        }

        # Time XGBoost
        if include_xgb:
            if warmup_key not in warmed_up_xgb:
                _warmup_xgboost(X, y, task, tree["max_depth"])
                warmed_up_xgb.add(warmup_key)

            xgb_times = _time_xgboost(X, y, task, tree["n_trees"], tree["max_depth"], trials)
            xgb_median = _median(xgb_times)
            speedup = xgb_median / ob_median if ob_median > 0 else 0
            entry["xgb_median_s"] = xgb_median
            entry["xgb_times"] = xgb_times
            entry["speedup"] = speedup
            print(f"    XGB: {xgb_median:.3f}s  speedup: {speedup:.2f}x")

        results[config_name] = entry

    # Compute score
    score, gates_passed = compute_speed_score(results)

    # Compute geo mean speedup for reporting
    speedups = [r["speedup"] for r in results.values() if r["speedup"] is not None]
    import math
    geo_mean_speedup = (
        math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        if speedups else None
    )

    return {
        "score": score,
        "gates_passed": gates_passed,
        "configs": results,
        "geo_mean_speedup": geo_mean_speedup,
    }
