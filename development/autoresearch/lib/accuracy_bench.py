"""Accuracy benchmarks for autoresearch v2.

Compares OpenBoost vs XGBoost on synthetic datasets,
computing parity ratios for each metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .data_generators import (
    make_accuracy_binary,
    make_accuracy_multiclass,
    make_accuracy_poisson,
    make_accuracy_regression,
)
from .scoring import compute_accuracy_score


ACCURACY_DATASETS = [
    {"name": "realistic_reg", "task": "regression"},
    {"name": "realistic_bin", "task": "binary"},
    {"name": "realistic_multi", "task": "multiclass"},
    {"name": "poisson", "task": "poisson"},
]

QUICK_DATASETS = ["realistic_reg"]

# Shared hyperparams for accuracy benchmarks
ACC_N_TREES = 200
ACC_MAX_DEPTH = 6
ACC_LR = 0.1


def _make_data(dataset_name: str):
    """Generate train/test data for a dataset."""
    if dataset_name == "realistic_reg":
        return make_accuracy_regression()
    elif dataset_name == "realistic_bin":
        return make_accuracy_binary()
    elif dataset_name == "realistic_multi":
        return make_accuracy_multiclass()
    elif dataset_name == "poisson":
        return make_accuracy_poisson()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def _train_openboost(X_train, y_train, task: str):
    """Train an OpenBoost model and return it."""
    import openboost as ob

    if task == "multiclass":
        n_classes = len(np.unique(y_train))
        model = ob.MultiClassGradientBoosting(
            n_classes=n_classes, n_trees=ACC_N_TREES,
            max_depth=ACC_MAX_DEPTH, learning_rate=ACC_LR,
        )
    elif task == "binary":
        model = ob.GradientBoosting(
            n_trees=ACC_N_TREES, max_depth=ACC_MAX_DEPTH,
            learning_rate=ACC_LR, loss="logloss",
        )
    elif task == "poisson":
        model = ob.GradientBoosting(
            n_trees=ACC_N_TREES, max_depth=ACC_MAX_DEPTH,
            learning_rate=ACC_LR, loss="poisson",
        )
    else:
        model = ob.GradientBoosting(
            n_trees=ACC_N_TREES, max_depth=ACC_MAX_DEPTH,
            learning_rate=ACC_LR, loss="mse",
        )

    model.fit(X_train, y_train)
    return model


def _train_xgboost(X_train, y_train, task: str):
    """Train an XGBoost model and return it."""
    import xgboost as xgb

    if task == "multiclass":
        model = xgb.XGBClassifier(
            n_estimators=ACC_N_TREES, max_depth=ACC_MAX_DEPTH,
            learning_rate=ACC_LR, tree_method="hist",
            device="cuda", max_bin=256, verbosity=0,
        )
    elif task == "binary":
        model = xgb.XGBClassifier(
            n_estimators=ACC_N_TREES, max_depth=ACC_MAX_DEPTH,
            learning_rate=ACC_LR, tree_method="hist",
            device="cuda", max_bin=256, verbosity=0,
            objective="binary:logistic",
        )
    elif task == "poisson":
        model = xgb.XGBRegressor(
            n_estimators=ACC_N_TREES, max_depth=ACC_MAX_DEPTH,
            learning_rate=ACC_LR, tree_method="hist",
            device="cuda", max_bin=256, verbosity=0,
            objective="count:poisson",
        )
    else:
        model = xgb.XGBRegressor(
            n_estimators=ACC_N_TREES, max_depth=ACC_MAX_DEPTH,
            learning_rate=ACC_LR, tree_method="hist",
            device="cuda", max_bin=256, verbosity=0,
        )

    model.fit(X_train, y_train)
    return model


def _compute_metrics(model, X_test, y_test, task: str, framework: str) -> dict:
    """Compute task-appropriate metrics."""
    if task == "regression":
        pred = model.predict(X_test)
        mse = float(np.mean((pred - y_test) ** 2))
        rmse = float(np.sqrt(mse))
        ss_res = np.sum((pred - y_test) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
        return {"rmse": rmse, "r2": r2, "primary": r2, "primary_name": "r2", "higher_is_better": True}

    elif task == "binary":
        if framework == "ob":
            raw = model.predict(X_test)
            # GradientBoosting with logloss returns log-odds; apply sigmoid
            pred = 1.0 / (1.0 + np.exp(-np.clip(raw, -30, 30)))
            pred_class = (pred > 0.5).astype(np.float32)
        else:
            pred = model.predict_proba(X_test)[:, 1]
            pred_class = model.predict(X_test)
        accuracy = float(np.mean(pred_class == y_test))
        # Simple AUC approximation via sorted thresholds
        auc = _roc_auc(y_test, pred)
        return {"accuracy": accuracy, "auc": auc, "primary": auc, "primary_name": "auc", "higher_is_better": True}

    elif task == "multiclass":
        if framework == "ob":
            pred_class = model.predict(X_test)
        else:
            pred_class = model.predict(X_test)
        accuracy = float(np.mean(pred_class == y_test))
        return {"accuracy": accuracy, "primary": accuracy, "primary_name": "accuracy", "higher_is_better": True}

    elif task == "poisson":
        raw_pred = model.predict(X_test)
        if framework == "ob":
            # OpenBoost Poisson loss works in log-space; convert to mu
            pred = np.exp(np.clip(raw_pred, -20, 20))
        else:
            # XGBoost count:poisson returns mu directly
            pred = raw_pred
        pred = np.maximum(pred, 1e-6)
        # Poisson deviance: 2 * sum(y*log(y/mu) - (y - mu))
        deviance = float(2 * np.mean(
            np.where(y_test > 0, y_test * np.log(np.maximum(y_test, 1e-6) / pred), 0) - (y_test - pred)
        ))
        return {"deviance": deviance, "primary": deviance, "primary_name": "deviance", "higher_is_better": False}

    raise ValueError(f"Unknown task: {task}")


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Simple ROC AUC computation without sklearn."""
    desc_indices = np.argsort(-y_score)
    y_sorted = y_true[desc_indices].astype(bool)
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg
    # Trapezoidal integration (np.trapezoid in numpy 2.0+, np.trapz before)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    auc = float(_trapz(tpr, fpr))
    return auc


def run_accuracy_benchmarks(
    include_xgb: bool = True,
    quick: bool = False,
) -> dict[str, Any]:
    """Run accuracy benchmarks across datasets.

    Returns dict with:
        score: float in [0, 1]
        gates_passed: bool
        datasets: dict[name, {ob_metrics, xgb_metrics, parity}]
        geo_mean_parity: float
    """
    import math

    datasets = QUICK_DATASETS if quick else [d["name"] for d in ACCURACY_DATASETS]
    results: dict[str, dict] = {}

    for ds_info in ACCURACY_DATASETS:
        ds_name = ds_info["name"]
        if ds_name not in datasets:
            continue

        task = ds_info["task"]
        print(f"\n  [accuracy] {ds_name} ({task})...")

        X_train, X_test, y_train, y_test = _make_data(ds_name)

        # Train OpenBoost
        ob_model = _train_openboost(X_train, y_train, task)
        ob_metrics = _compute_metrics(ob_model, X_test, y_test, task, "ob")
        print(f"    OB {ob_metrics['primary_name']}: {ob_metrics['primary']:.4f}")

        entry = {
            "task": task,
            "ob_metrics": ob_metrics,
            "xgb_metrics": None,
            "parity": 1.0,  # default if no XGBoost
        }

        # Train XGBoost for comparison
        if include_xgb:
            xgb_model = _train_xgboost(X_train, y_train, task)
            xgb_metrics = _compute_metrics(xgb_model, X_test, y_test, task, "xgb")
            entry["xgb_metrics"] = xgb_metrics
            print(f"    XGB {xgb_metrics['primary_name']}: {xgb_metrics['primary']:.4f}")

            # Parity: >1 means OB is better
            if ob_metrics["higher_is_better"]:
                parity = ob_metrics["primary"] / xgb_metrics["primary"] if xgb_metrics["primary"] > 0 else 1.0
            else:
                parity = xgb_metrics["primary"] / ob_metrics["primary"] if ob_metrics["primary"] > 0 else 1.0

            entry["parity"] = float(parity)
            print(f"    parity: {parity:.3f}")

        results[ds_name] = entry

    # Compute score
    score, gates_passed = compute_accuracy_score(results)

    # Geo mean parity
    parities = [r["parity"] for r in results.values()]
    geo_mean_parity = (
        math.exp(sum(math.log(p) for p in parities) / len(parities))
        if parities else 1.0
    )

    return {
        "score": score,
        "gates_passed": gates_passed,
        "datasets": results,
        "geo_mean_parity": geo_mean_parity,
    }
