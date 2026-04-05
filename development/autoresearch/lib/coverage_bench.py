"""Feature coverage benchmarks for autoresearch v2.

Tests that model variants and feature types work correctly.
Each test returns {passed, metrics, error}.
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np

from .data_generators import make_categorical_data, make_missing_data, make_small_regression
from .scoring import compute_coverage_score


# ---------------------------------------------------------------------------
# Individual coverage tests
# ---------------------------------------------------------------------------

def _test_missing_values() -> dict:
    """Train on data with 5% NaN, verify model learns (R2 > 0.5)."""
    import openboost as ob

    X_train, X_test, y_train, y_test = make_missing_data(
        n_samples=200_000, n_features=50, missing_rate=0.05
    )

    model = ob.GradientBoosting(
        n_trees=200, max_depth=6, learning_rate=0.1, loss="mse"
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = float(np.mean((pred - y_test) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float(1 - np.sum((pred - y_test) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    # Gate: model must learn meaningfully despite missing data
    passed = r2 > 0.5 and np.isfinite(mse)

    return {
        "passed": passed,
        "metrics": {"mse": mse, "r2": r2},
        "error": None if passed else f"R2 {r2:.4f} <= 0.5 (MSE={mse:.4f})",
    }


def _test_categoricals() -> dict:
    """Train with categorical features, verify no crash and reasonable accuracy."""
    import openboost as ob

    X_train, X_test, y_train, y_test, cat_indices = make_categorical_data(
        n_samples=50_000, n_numeric=10, n_categorical=5
    )

    ba = ob.array(X_train, categorical_features=cat_indices)
    model = ob.GradientBoosting(
        n_trees=100, max_depth=6, learning_rate=0.1, loss="mse"
    )
    model.fit(ba, y_train)
    pred = model.predict(X_test)

    mse = float(np.mean((pred - y_test) ** 2))
    r2 = float(1 - np.sum((pred - y_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))

    return {
        "passed": np.isfinite(mse) and r2 > 0,
        "metrics": {"mse": mse, "r2": r2},
        "error": None,
    }


def _test_naturalboost() -> dict:
    """Train NaturalBoostNormal, verify NLL is finite."""
    import openboost as ob

    X_train, X_test, y_train, y_test = make_small_regression()

    model = ob.NaturalBoostNormal(
        n_trees=50, max_depth=3, learning_rate=0.1
    )
    model.fit(X_train, y_train)

    nll = float(model.nll(X_test, y_test))
    pred_mean = model.predict(X_test)
    mse = float(np.mean((pred_mean - y_test) ** 2))

    return {
        "passed": np.isfinite(nll) and np.isfinite(mse),
        "metrics": {"nll": nll, "mse": mse},
        "error": None if np.isfinite(nll) else "NLL is not finite",
    }


def _test_dart() -> dict:
    """Train DART model, verify MSE decreases vs untrained."""
    import openboost as ob

    X_train, X_test, y_train, y_test = make_small_regression()

    model = ob.DART(
        n_trees=50, max_depth=4, learning_rate=0.1, dropout_rate=0.1
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = float(np.mean((pred - y_test) ** 2))
    baseline_mse = float(np.mean((y_test - np.mean(y_train)) ** 2))
    improvement = 1.0 - mse / baseline_mse if baseline_mse > 0 else 0.0

    return {
        "passed": mse < baseline_mse and np.isfinite(mse),
        "metrics": {"mse": mse, "baseline_mse": baseline_mse, "improvement": improvement},
        "error": None if mse < baseline_mse else f"MSE {mse:.4f} >= baseline {baseline_mse:.4f}",
    }


def _test_gam() -> dict:
    """Train OpenBoostGAM, verify R2 > 0."""
    import openboost as ob

    X_train, X_test, y_train, y_test = make_small_regression()

    model = ob.OpenBoostGAM(
        n_rounds=500, learning_rate=0.05
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = float(np.mean((pred - y_test) ** 2))
    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
    r2 = float(1 - np.sum((pred - y_test) ** 2) / ss_tot) if ss_tot > 0 else 0.0

    return {
        "passed": r2 > 0 and np.isfinite(r2),
        "metrics": {"mse": mse, "r2": r2},
        "error": None if r2 > 0 else f"R2 {r2:.4f} <= 0",
    }


def _test_growth_strategy(strategy: str) -> dict:
    """Train using fit_tree with a specific growth strategy, verify it works."""
    import openboost as ob
    from openboost._core._tree import fit_tree
    from openboost._loss import get_loss_function

    X_train, X_test, y_train, y_test = make_small_regression()

    ba = ob.array(X_train)
    loss_fn = get_loss_function("mse")
    n_trees = 50
    lr = 0.1
    pred_train = np.zeros(len(y_train), dtype=np.float32)

    # Move to GPU if CUDA is active
    if ob.is_cuda():
        from numba import cuda
        pred_train = cuda.to_device(pred_train)
        y_gpu = cuda.to_device(y_train)

    trees = []
    for _ in range(n_trees):
        if ob.is_cuda():
            grad, hess = loss_fn(pred_train, y_gpu)
        else:
            grad, hess = loss_fn(pred_train, y_train)
        tree = fit_tree(
            ba, grad, hess,
            max_depth=6, reg_lambda=1.0,
            growth=strategy,
        )
        trees.append(tree)
        tree_pred = tree(ba)
        if ob.is_cuda():
            # Both should be GPU arrays; use CUDA kernel for in-place add
            from openboost._core._predict import _add_inplace_cuda
            _add_inplace_cuda(pred_train, tree_pred, lr)
        else:
            if hasattr(tree_pred, 'copy_to_host'):
                tree_pred = tree_pred.copy_to_host()
            pred_train += lr * tree_pred

    # Evaluate on training set
    if hasattr(pred_train, 'copy_to_host'):
        pred_np = pred_train.copy_to_host()
    else:
        pred_np = pred_train

    mse = float(np.mean((pred_np - y_train) ** 2))
    baseline_mse = float(np.mean((y_train - np.mean(y_train)) ** 2))

    return {
        "passed": mse < baseline_mse and np.isfinite(mse),
        "metrics": {"mse": mse, "baseline_mse": baseline_mse},
        "error": None if mse < baseline_mse else f"MSE {mse:.4f} >= baseline {baseline_mse:.4f}",
    }


def _test_symmetric_growth() -> dict:
    """Train with symmetric growth strategy, verify no crash."""
    return _test_growth_strategy("symmetric")


def _test_leafwise_growth() -> dict:
    """Train with leaf-wise growth strategy, verify no crash."""
    return _test_growth_strategy("leafwise")


# ---------------------------------------------------------------------------
# Test registry
# ---------------------------------------------------------------------------

COVERAGE_TESTS = {
    "missing_values": _test_missing_values,
    "categoricals": _test_categoricals,
    "naturalboost": _test_naturalboost,
    "dart": _test_dart,
    "gam": _test_gam,
    "symmetric_growth": _test_symmetric_growth,
    "leafwise_growth": _test_leafwise_growth,
}

QUICK_TESTS = ["missing_values", "categoricals"]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_coverage_benchmarks(quick: bool = False) -> dict[str, Any]:
    """Run all coverage tests.

    Returns dict with:
        score: float in [0, 1]
        gates_passed: bool
        tests: dict[name, {passed, metrics, error}]
        passed_count: int
        total_count: int
    """
    test_names = QUICK_TESTS if quick else list(COVERAGE_TESTS.keys())
    results: dict[str, dict] = {}

    for name in test_names:
        print(f"\n  [coverage] {name}...")
        test_fn = COVERAGE_TESTS[name]
        try:
            result = test_fn()
        except Exception as e:
            result = {
                "passed": False,
                "metrics": {},
                "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            }

        status = "PASS" if result["passed"] else "FAIL"
        print(f"    {status}", end="")
        if result.get("metrics"):
            metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in result["metrics"].items()
                                   if isinstance(v, (int, float)))
            print(f" ({metrics_str})", end="")
        if result.get("error"):
            print(f" -- {result['error']}", end="")
        print()

        results[name] = result

    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)

    score, gates_passed = compute_coverage_score(results)

    return {
        "score": score,
        "gates_passed": gates_passed,
        "tests": results,
        "passed_count": passed_count,
        "total_count": total_count,
    }
