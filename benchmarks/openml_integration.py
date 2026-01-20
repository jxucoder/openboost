"""Phase 19: OpenML Integration Tests.

Pre-release validation of OpenBoost vs XGBoost performance on real-world datasets.

Run on Modal (GPU):
    uv run modal run benchmarks/openml_integration.py

Run locally (small datasets only):
    uv run python benchmarks/openml_integration.py --local

Run specific datasets:
    uv run modal run benchmarks/openml_integration.py --datasets cpu_act higgs

Run specific configs:
    uv run modal run benchmarks/openml_integration.py --configs baseline deep_tree

Run extended suite:
    uv run modal run benchmarks/openml_integration.py --extended
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import modal
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# Modal Configuration
# =============================================================================

app = modal.App("openboost-integration-tests")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.12")
    .pip_install(
        "numpy>=1.24",
        "numba>=0.60",
        "scikit-learn>=1.0",
        "xgboost>=2.0",
        "openml>=0.14",
        "pandas>=2.0",
        "tabulate>=0.9",
    )
    .add_local_dir(
        str(PROJECT_ROOT / "src" / "openboost"),
        remote_path="/root/openboost",
    )
)

# =============================================================================
# Dataset Definitions
# =============================================================================

# Primary datasets - must pass for release
PRIMARY_DATASETS = [
    {"name": "cpu_act", "id": 44, "task": "regression"},
    {"name": "higgs", "id": 44092, "task": "binary"},  # HIGGS binary classification
    {"name": "covertype", "id": 1596, "task": "multiclass"},  # Forest covertype 7-class
]

# Extended datasets - should pass for full validation
EXTENDED_DATASETS = [
    {"name": "pol", "id": 201, "task": "regression"},
    {"name": "house_16H", "id": 216, "task": "regression"},
    {"name": "wine_quality", "id": 287, "task": "multiclass"},
    {"name": "bank-marketing", "id": 1510, "task": "binary"},
    {"name": "Bioresponse", "id": 4134, "task": "binary"},
    {"name": "albert", "id": 41143, "task": "binary"},
    {"name": "BNG_cpu_act", "id": 42705, "task": "regression"},
]

# All datasets
ALL_DATASETS = PRIMARY_DATASETS + EXTENDED_DATASETS

# Dataset lookup by name
DATASET_BY_NAME = {d["name"]: d for d in ALL_DATASETS}


# =============================================================================
# Hyperparameter Configurations
# =============================================================================

HYPERPARAMETER_CONFIGS = {
    "baseline": {
        "n_trees": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 1.0,
        "min_child_weight": 1,
        "reg_lambda": 1,
        "description": "Default configuration",
    },
    "low_lr": {
        "n_trees": 500,
        "max_depth": 6,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 1,
        "description": "Low learning rate + many trees",
    },
    "deep_tree": {
        "n_trees": 100,
        "max_depth": 10,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 5,
        "description": "Deep trees with regularization",
    },
    "shallow_fast": {
        "n_trees": 200,
        "max_depth": 4,
        "learning_rate": 0.3,
        "subsample": 0.8,
        "min_child_weight": 1,
        "reg_lambda": 0,
        "description": "Shallow trees + aggressive learning rate",
    },
    "regularized": {
        "n_trees": 200,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.5,
        "min_child_weight": 10,
        "reg_lambda": 5,
        "description": "Heavy regularization",
    },
    "goss": {
        "n_trees": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample_strategy": "goss",
        "goss_top_rate": 0.2,
        "goss_other_rate": 0.1,
        "description": "GOSS sampling (OpenBoost-specific)",
    },
}

DEFAULT_CONFIGS = ["baseline", "deep_tree", "regularized", "shallow_fast"]


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    model_name: str
    train_time: float  # seconds
    pred_time: float  # milliseconds
    metrics: dict[str, float]


@dataclass
class ConfigResult:
    """Results for a single config on a single dataset."""

    dataset_name: str
    config_name: str
    task_type: str
    n_samples: int
    n_features: int
    openboost: BenchmarkResult
    xgboost: BenchmarkResult
    passed: bool
    failure_reason: str | None = None


@dataclass
class DatasetResult:
    """Aggregated results for a single dataset."""

    dataset_name: str
    task_type: str
    n_samples: int
    n_features: int
    config_results: list[ConfigResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.config_results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.config_results if not r.passed)

    @property
    def pass_rate(self) -> float:
        if not self.config_results:
            return 0.0
        return self.passed_count / len(self.config_results)


# =============================================================================
# Dataset Loader
# =============================================================================


def load_openml_dataset(
    dataset_id: int, max_samples: int | None = None, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load dataset from OpenML.

    Args:
        dataset_id: OpenML dataset ID
        max_samples: Maximum samples to load (for testing)
        verbose: Print loading info

    Returns:
        X: Feature array (float32)
        y: Target array
        task_type: 'regression', 'binary', or 'multiclass'
    """
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder

    if verbose:
        print(f"Loading OpenML dataset {dataset_id}...")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = fetch_openml(data_id=dataset_id, as_frame=False, parser="auto")

    X = dataset.data
    y = dataset.target

    # Handle feature types
    if hasattr(X, "toarray"):  # Sparse matrix
        X = X.toarray()

    # Handle object dtype features (convert to numeric)
    if X.dtype == object:
        from sklearn.preprocessing import OrdinalEncoder

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X = encoder.fit_transform(X)

    X = X.astype(np.float32)

    # Handle NaN values
    if np.any(np.isnan(X)):
        from sklearn.impute import SimpleImputer

        imputer = SimpleImputer(strategy="median")
        X = imputer.fit_transform(X)

    # Determine task type
    if y.dtype in [np.float64, np.float32] or (
        y.dtype == object
        and all(
            str(val).replace(".", "", 1).replace("-", "", 1).isdigit() for val in y[:100]
        )
    ):
        # Regression
        y = y.astype(np.float32)
        task_type = "regression"
    else:
        # Classification
        le = LabelEncoder()
        y = le.fit_transform(y).astype(np.int32)
        n_classes = len(np.unique(y))
        task_type = "binary" if n_classes == 2 else "multiclass"

    # Subsample if needed
    if max_samples is not None and len(X) > max_samples:
        indices = np.random.RandomState(42).choice(len(X), max_samples, replace=False)
        X = X[indices]
        y = y[indices]

    if verbose:
        print(
            f"  Loaded: {X.shape[0]:,} samples × {X.shape[1]} features, task={task_type}"
        )

    return X, y, task_type


# =============================================================================
# Metrics
# =============================================================================


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray | None,
    task_type: str,
) -> dict[str, float]:
    """Compute evaluation metrics.

    Args:
        y_true: Ground truth
        y_pred: Predictions (class labels for classification)
        y_pred_proba: Predicted probabilities (for classification)
        task_type: 'regression', 'binary', or 'multiclass'

    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
        roc_auc_score,
    )

    metrics = {}

    if task_type == "regression":
        metrics["rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["r2"] = r2_score(y_true, y_pred)
        metrics["mae"] = mean_absolute_error(y_true, y_pred)
    elif task_type == "binary":
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        if y_pred_proba is not None:
            try:
                metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
                metrics["logloss"] = log_loss(y_true, y_pred_proba)
            except Exception:
                pass
    else:  # multiclass
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["f1_macro"] = f1_score(y_true, y_pred, average="macro")
        if y_pred_proba is not None:
            try:
                metrics["logloss"] = log_loss(y_true, y_pred_proba)
            except Exception:
                pass

    return metrics


def check_acceptance_criteria(
    ob_metrics: dict[str, float],
    xgb_metrics: dict[str, float],
    task_type: str,
) -> tuple[bool, str | None]:
    """Check if OpenBoost meets acceptance criteria vs XGBoost.

    Acceptance criteria:
    - Regression: RMSE within ±5% of XGBoost
    - Binary Classification: AUC within ±0.01 of XGBoost
    - Multi-class: Accuracy within ±2% of XGBoost

    Args:
        ob_metrics: OpenBoost metrics
        xgb_metrics: XGBoost metrics
        task_type: Task type

    Returns:
        (passed, failure_reason)
    """
    if task_type == "regression":
        ob_rmse = ob_metrics.get("rmse", float("inf"))
        xgb_rmse = xgb_metrics.get("rmse", float("inf"))
        ratio = ob_rmse / xgb_rmse if xgb_rmse > 0 else float("inf")
        if ratio > 1.05:
            return False, f"RMSE {ratio:.2%} of XGBoost (limit: 105%)"
        return True, None

    elif task_type == "binary":
        ob_auc = ob_metrics.get("auc", 0)
        xgb_auc = xgb_metrics.get("auc", 1)
        diff = ob_auc - xgb_auc
        if diff < -0.01:
            return False, f"AUC {diff:+.4f} vs XGBoost (limit: -0.01)"
        return True, None

    else:  # multiclass
        ob_acc = ob_metrics.get("accuracy", 0)
        xgb_acc = xgb_metrics.get("accuracy", 1)
        diff = (ob_acc - xgb_acc) * 100  # percentage points
        if diff < -2:
            return False, f"Accuracy {diff:+.1f}pp vs XGBoost (limit: -2pp)"
        return True, None


# =============================================================================
# Model Runners
# =============================================================================


def run_openboost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    config: dict[str, Any],
    use_gpu: bool = True,
) -> BenchmarkResult:
    """Run OpenBoost model.

    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Target arrays
        task_type: Task type
        config: Hyperparameter configuration
        use_gpu: Whether running on GPU

    Returns:
        BenchmarkResult
    """
    import openboost as ob

    n_trees = config.get("n_trees", 100)
    max_depth = config.get("max_depth", 6)
    learning_rate = config.get("learning_rate", 0.1)
    subsample = config.get("subsample", 1.0)
    min_child_weight = config.get("min_child_weight", 1)
    reg_lambda = config.get("reg_lambda", 1)

    # Handle GOSS config - pass parameters directly
    subsample_strategy = config.get("subsample_strategy", "none")
    goss_top_rate = config.get("goss_top_rate", 0.2)
    goss_other_rate = config.get("goss_other_rate", 0.1)
    if subsample_strategy == "goss":
        subsample = 1.0  # GOSS handles sampling

    # Warmup JIT compilation
    if task_type == "multiclass":
        n_classes = len(np.unique(y_train))
        warmup_model = ob.MultiClassGradientBoosting(
            n_classes=n_classes, n_trees=2, max_depth=3
        )
    else:
        loss = "logloss" if task_type == "binary" else "mse"
        warmup_model = ob.GradientBoosting(n_trees=2, max_depth=3, loss=loss)

    warmup_model.fit(X_train[:min(500, len(X_train))], y_train[:min(500, len(y_train))])

    if use_gpu:
        from numba import cuda

        cuda.synchronize()

    # Create and train model
    if task_type == "multiclass":
        n_classes = len(np.unique(y_train))
        model = ob.MultiClassGradientBoosting(
            n_classes=n_classes,
            n_trees=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            subsample_strategy=subsample_strategy,
            goss_top_rate=goss_top_rate,
            goss_other_rate=goss_other_rate,
        )
    else:
        loss = "logloss" if task_type == "binary" else "mse"
        model = ob.GradientBoosting(
            n_trees=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate,
            loss=loss,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            subsample_strategy=subsample_strategy,
            goss_top_rate=goss_top_rate,
            goss_other_rate=goss_other_rate,
        )

    # Training
    start = time.perf_counter()
    model.fit(X_train, y_train)
    if use_gpu:
        from numba import cuda

        cuda.synchronize()
    train_time = time.perf_counter() - start

    # Prediction
    start = time.perf_counter()
    if task_type == "multiclass":
        y_pred_proba = model.predict_proba(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
    elif task_type == "binary":
        y_pred_raw = model.predict(X_test)
        y_pred_proba = 1 / (1 + np.exp(-y_pred_raw))
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:  # regression
        y_pred = model.predict(X_test)
        y_pred_proba = None

    if use_gpu:
        from numba import cuda

        cuda.synchronize()
    pred_time = (time.perf_counter() - start) * 1000  # Convert to ms

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_pred_proba, task_type)

    return BenchmarkResult(
        model_name="OpenBoost",
        train_time=train_time,
        pred_time=pred_time,
        metrics=metrics,
    )


def run_xgboost(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    config: dict[str, Any],
    use_gpu: bool = True,
) -> BenchmarkResult:
    """Run XGBoost model.

    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Target arrays
        task_type: Task type
        config: Hyperparameter configuration
        use_gpu: Whether running on GPU

    Returns:
        BenchmarkResult
    """
    import xgboost as xgb

    n_trees = config.get("n_trees", 100)
    max_depth = config.get("max_depth", 6)
    learning_rate = config.get("learning_rate", 0.1)
    subsample = config.get("subsample", 1.0)
    min_child_weight = config.get("min_child_weight", 1)
    reg_lambda = config.get("reg_lambda", 1)

    # Skip GOSS for XGBoost (not directly supported)
    if config.get("subsample_strategy") == "goss":
        subsample = 0.3  # Approximate similar sample rate

    device = "cuda" if use_gpu else "cpu"

    # Create model
    if task_type == "multiclass":
        n_classes = len(np.unique(y_train))
        model = xgb.XGBClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            tree_method="hist",
            device=device,
            objective="multi:softprob",
            num_class=n_classes,
            verbosity=0,
        )
    elif task_type == "binary":
        model = xgb.XGBClassifier(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            verbosity=0,
        )
    else:  # regression
        model = xgb.XGBRegressor(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            min_child_weight=min_child_weight,
            reg_lambda=reg_lambda,
            tree_method="hist",
            device=device,
            verbosity=0,
        )

    # Training
    start = time.perf_counter()
    model.fit(X_train, y_train)
    if use_gpu:
        from numba import cuda

        cuda.synchronize()
    train_time = time.perf_counter() - start

    # Prediction
    start = time.perf_counter()
    if task_type in ["binary", "multiclass"]:
        y_pred_proba = model.predict_proba(X_test)
        if task_type == "binary":
            y_pred_proba = y_pred_proba[:, 1]
        y_pred = model.predict(X_test)
    else:  # regression
        y_pred = model.predict(X_test)
        y_pred_proba = None

    if use_gpu:
        from numba import cuda

        cuda.synchronize()
    pred_time = (time.perf_counter() - start) * 1000  # Convert to ms

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_pred_proba, task_type)

    return BenchmarkResult(
        model_name="XGBoost",
        train_time=train_time,
        pred_time=pred_time,
        metrics=metrics,
    )


# =============================================================================
# Benchmark Orchestration
# =============================================================================


def benchmark_single_config(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    task_type: str,
    dataset_name: str,
    config_name: str,
    config: dict[str, Any],
    use_gpu: bool = True,
) -> ConfigResult:
    """Run benchmark for a single configuration.

    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Target arrays
        task_type: Task type
        dataset_name: Name of the dataset
        config_name: Name of the configuration
        config: Hyperparameter configuration
        use_gpu: Whether running on GPU

    Returns:
        ConfigResult
    """
    print(f"\n  Config: {config_name}")
    print(f"    {config.get('description', '')}")

    # Run OpenBoost
    try:
        ob_result = run_openboost(
            X_train, X_test, y_train, y_test, task_type, config, use_gpu
        )
        print(f"    OpenBoost: {ob_result.train_time:.2f}s train, {ob_result.pred_time:.1f}ms pred")
    except Exception as e:
        print(f"    OpenBoost FAILED: {e}")
        ob_result = BenchmarkResult(
            model_name="OpenBoost",
            train_time=float("inf"),
            pred_time=float("inf"),
            metrics={},
        )

    # Run XGBoost
    try:
        xgb_result = run_xgboost(
            X_train, X_test, y_train, y_test, task_type, config, use_gpu
        )
        print(f"    XGBoost:   {xgb_result.train_time:.2f}s train, {xgb_result.pred_time:.1f}ms pred")
    except Exception as e:
        print(f"    XGBoost FAILED: {e}")
        xgb_result = BenchmarkResult(
            model_name="XGBoost",
            train_time=float("inf"),
            pred_time=float("inf"),
            metrics={},
        )

    # Check acceptance criteria
    passed, failure_reason = check_acceptance_criteria(
        ob_result.metrics, xgb_result.metrics, task_type
    )

    status = "✓ PASS" if passed else f"✗ FAIL: {failure_reason}"
    print(f"    Result: {status}")

    return ConfigResult(
        dataset_name=dataset_name,
        config_name=config_name,
        task_type=task_type,
        n_samples=len(X_train) + len(X_test),
        n_features=X_train.shape[1],
        openboost=ob_result,
        xgboost=xgb_result,
        passed=passed,
        failure_reason=failure_reason,
    )


def benchmark_dataset(
    dataset_info: dict[str, Any],
    configs: list[str],
    use_gpu: bool = True,
    max_samples: int | None = None,
) -> DatasetResult:
    """Run all configs on a single dataset.

    Args:
        dataset_info: Dataset info dict with name, id, task
        configs: List of config names to run
        use_gpu: Whether running on GPU
        max_samples: Maximum samples (for testing)

    Returns:
        DatasetResult
    """
    from sklearn.model_selection import train_test_split

    dataset_name = dataset_info["name"]
    dataset_id = dataset_info["id"]
    expected_task = dataset_info["task"]

    print(f"\n{'=' * 60}")
    print(f"Dataset: {dataset_name} (ID: {dataset_id})")
    print(f"{'=' * 60}")

    # Load data
    try:
        X, y, task_type = load_openml_dataset(dataset_id, max_samples=max_samples)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return DatasetResult(
            dataset_name=dataset_name,
            task_type=expected_task,
            n_samples=0,
            n_features=0,
        )

    # Verify task type
    if task_type != expected_task:
        print(f"Warning: Expected task '{expected_task}', got '{task_type}'")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task_type != "regression" else None
    )

    print(f"Task: {task_type}")
    print(f"Train: {X_train.shape[0]:,} samples × {X_train.shape[1]} features")
    print(f"Test:  {X_test.shape[0]:,} samples")

    # Run configs
    result = DatasetResult(
        dataset_name=dataset_name,
        task_type=task_type,
        n_samples=len(X),
        n_features=X.shape[1],
    )

    for config_name in configs:
        if config_name not in HYPERPARAMETER_CONFIGS:
            print(f"  Unknown config: {config_name}, skipping")
            continue

        config = HYPERPARAMETER_CONFIGS[config_name]
        config_result = benchmark_single_config(
            X_train,
            X_test,
            y_train,
            y_test,
            task_type,
            dataset_name,
            config_name,
            config,
            use_gpu,
        )
        result.config_results.append(config_result)

    return result


# =============================================================================
# Results Formatting
# =============================================================================


def format_results_table(results: list[DatasetResult]) -> str:
    """Format results as a pretty table.

    Args:
        results: List of DatasetResult

    Returns:
        Formatted string
    """
    try:
        from tabulate import tabulate
    except ImportError:
        tabulate = None

    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("DETAILED RESULTS")
    lines.append("=" * 70)

    for dataset_result in results:
        lines.append(f"\nDataset: {dataset_result.dataset_name}")
        lines.append(f"  Samples: {dataset_result.n_samples:,}")
        lines.append(f"  Features: {dataset_result.n_features}")
        lines.append(f"  Task: {dataset_result.task_type}")
        lines.append("")

        # Build table data
        table_data = []
        headers = ["Config", "Model", "Train(s)", "Pred(ms)", "Primary Metric", "Status"]

        for config_result in dataset_result.config_results:
            # Get primary metric name
            if config_result.task_type == "regression":
                metric_name = "RMSE"
                ob_metric = config_result.openboost.metrics.get("rmse", float("nan"))
                xgb_metric = config_result.xgboost.metrics.get("rmse", float("nan"))
            elif config_result.task_type == "binary":
                metric_name = "AUC"
                ob_metric = config_result.openboost.metrics.get("auc", float("nan"))
                xgb_metric = config_result.xgboost.metrics.get("auc", float("nan"))
            else:
                metric_name = "Accuracy"
                ob_metric = config_result.openboost.metrics.get("accuracy", float("nan"))
                xgb_metric = config_result.xgboost.metrics.get("accuracy", float("nan"))

            # OpenBoost row
            status = "✓" if config_result.passed else "✗"
            table_data.append([
                config_result.config_name,
                "OpenBoost",
                f"{config_result.openboost.train_time:.2f}",
                f"{config_result.openboost.pred_time:.1f}",
                f"{ob_metric:.4f}",
                status,
            ])

            # XGBoost row
            table_data.append([
                "",
                "XGBoost",
                f"{config_result.xgboost.train_time:.2f}",
                f"{config_result.xgboost.pred_time:.1f}",
                f"{xgb_metric:.4f}",
                "(ref)",
            ])

            # Ratio row
            train_ratio = (
                config_result.openboost.train_time / config_result.xgboost.train_time
                if config_result.xgboost.train_time > 0
                else float("inf")
            )
            pred_ratio = (
                config_result.openboost.pred_time / config_result.xgboost.pred_time
                if config_result.xgboost.pred_time > 0
                else float("inf")
            )

            metric_diff = ob_metric - xgb_metric
            if config_result.task_type == "regression":
                metric_str = f"{(ob_metric / xgb_metric - 1) * 100:+.1f}%" if xgb_metric > 0 else "N/A"
            else:
                metric_str = f"{metric_diff:+.4f}"

            table_data.append([
                "",
                "Δ/Ratio",
                f"{train_ratio:.2f}x",
                f"{pred_ratio:.2f}x",
                metric_str,
                "",
            ])

        if tabulate:
            lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
        else:
            # Fallback formatting
            lines.append("  " + " | ".join(headers))
            lines.append("  " + "-" * 60)
            for row in table_data:
                lines.append("  " + " | ".join(str(c).ljust(12) for c in row))

    return "\n".join(lines)


def format_summary(results: list[DatasetResult]) -> str:
    """Format summary of all results.

    Args:
        results: List of DatasetResult

    Returns:
        Formatted string
    """
    lines = []
    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)

    total_passed = 0
    total_tests = 0

    table_data = []
    for dataset_result in results:
        passed = dataset_result.passed_count
        total = len(dataset_result.config_results)
        total_passed += passed
        total_tests += total

        # Average metric difference
        if dataset_result.task_type == "regression":
            diffs = []
            for cr in dataset_result.config_results:
                ob_rmse = cr.openboost.metrics.get("rmse", 0)
                xgb_rmse = cr.xgboost.metrics.get("rmse", 1)
                if xgb_rmse > 0:
                    diffs.append((ob_rmse / xgb_rmse - 1) * 100)
            avg_diff = f"{np.mean(diffs):+.1f}% RMSE" if diffs else "N/A"
        elif dataset_result.task_type == "binary":
            diffs = []
            for cr in dataset_result.config_results:
                ob_auc = cr.openboost.metrics.get("auc", 0)
                xgb_auc = cr.xgboost.metrics.get("auc", 0)
                diffs.append(ob_auc - xgb_auc)
            avg_diff = f"{np.mean(diffs):+.4f} AUC" if diffs else "N/A"
        else:
            diffs = []
            for cr in dataset_result.config_results:
                ob_acc = cr.openboost.metrics.get("accuracy", 0)
                xgb_acc = cr.xgboost.metrics.get("accuracy", 0)
                diffs.append((ob_acc - xgb_acc) * 100)
            avg_diff = f"{np.mean(diffs):+.1f}pp Acc" if diffs else "N/A"

        table_data.append([
            dataset_result.dataset_name,
            f"{passed}/{total}",
            passed,
            total - passed,
            avg_diff,
        ])

    headers = ["Dataset", "Pass Rate", "Passed", "Failed", "Avg Δ Metric"]

    try:
        from tabulate import tabulate

        lines.append(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        lines.append(" | ".join(headers))
        lines.append("-" * 60)
        for row in table_data:
            lines.append(" | ".join(str(c).ljust(12) for c in row))

    lines.append("")
    lines.append("=" * 70)
    pass_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    lines.append(f"OVERALL: {total_passed}/{total_tests} PASSED ({pass_rate:.1f}%)")
    lines.append("=" * 70)

    return "\n".join(lines)


def results_to_dict(results: list[DatasetResult]) -> dict[str, Any]:
    """Convert results to JSON-serializable dict.

    Args:
        results: List of DatasetResult

    Returns:
        Dictionary representation
    """
    output = {"datasets": []}

    for dataset_result in results:
        dataset_dict = {
            "name": dataset_result.dataset_name,
            "task_type": dataset_result.task_type,
            "n_samples": dataset_result.n_samples,
            "n_features": dataset_result.n_features,
            "passed": dataset_result.passed_count,
            "failed": dataset_result.failed_count,
            "configs": [],
        }

        for config_result in dataset_result.config_results:
            config_dict = {
                "name": config_result.config_name,
                "passed": config_result.passed,
                "failure_reason": config_result.failure_reason,
                "openboost": {
                    "train_time": config_result.openboost.train_time,
                    "pred_time": config_result.openboost.pred_time,
                    "metrics": config_result.openboost.metrics,
                },
                "xgboost": {
                    "train_time": config_result.xgboost.train_time,
                    "pred_time": config_result.xgboost.pred_time,
                    "metrics": config_result.xgboost.metrics,
                },
            }
            dataset_dict["configs"].append(config_dict)

        output["datasets"].append(dataset_dict)

    # Summary
    total_passed = sum(d["passed"] for d in output["datasets"])
    total_tests = sum(d["passed"] + d["failed"] for d in output["datasets"])
    output["summary"] = {
        "total_passed": total_passed,
        "total_tests": total_tests,
        "pass_rate": total_passed / total_tests if total_tests > 0 else 0,
    }

    return output


# =============================================================================
# Main Benchmark Runner
# =============================================================================


def run_integration_tests(
    datasets: list[str] | None = None,
    configs: list[str] | None = None,
    include_extended: bool = False,
    use_gpu: bool = True,
    max_samples: int | None = None,
) -> list[DatasetResult]:
    """Run full integration test suite.

    Args:
        datasets: List of dataset names (None = PRIMARY_DATASETS)
        configs: List of config names (None = DEFAULT_CONFIGS)
        include_extended: Include extended datasets
        use_gpu: Whether running on GPU
        max_samples: Maximum samples per dataset (for testing)

    Returns:
        List of DatasetResult
    """
    # Determine datasets
    if datasets is None:
        dataset_list = PRIMARY_DATASETS.copy()
        if include_extended:
            dataset_list.extend(EXTENDED_DATASETS)
    else:
        dataset_list = [DATASET_BY_NAME[name] for name in datasets if name in DATASET_BY_NAME]
        if not dataset_list:
            print(f"No valid datasets found. Available: {list(DATASET_BY_NAME.keys())}")
            return []

    # Determine configs
    if configs is None:
        configs = DEFAULT_CONFIGS

    print("=" * 70)
    print("OPENBOOST vs XGBOOST INTEGRATION TESTS")
    print("=" * 70)

    if use_gpu:
        try:
            from numba import cuda

            print(f"GPU: {cuda.get_current_device().name}")
        except Exception:
            print("GPU: Not available")
    else:
        print("Device: CPU")

    print(f"Datasets: {len(dataset_list)}")
    print(f"Configs: {configs}")
    if max_samples:
        print(f"Max samples: {max_samples:,}")
    print("")

    # Run benchmarks
    results = []
    for dataset_info in dataset_list:
        result = benchmark_dataset(
            dataset_info=dataset_info,
            configs=configs,
            use_gpu=use_gpu,
            max_samples=max_samples,
        )
        results.append(result)

    # Print results
    print(format_results_table(results))
    print(format_summary(results))

    return results


# =============================================================================
# Modal Entry Points
# =============================================================================


@app.function(gpu="A100", image=image, timeout=7200)
def benchmark_gpu(
    datasets: list[str] | None = None,
    configs: list[str] | None = None,
    include_extended: bool = False,
) -> dict[str, Any]:
    """Run benchmark on GPU (Modal).

    Args:
        datasets: List of dataset names (None = PRIMARY_DATASETS)
        configs: List of config names (None = DEFAULT_CONFIGS)
        include_extended: Include extended datasets

    Returns:
        Dictionary of results
    """
    import sys

    sys.path.insert(0, "/root")

    results = run_integration_tests(
        datasets=datasets,
        configs=configs,
        include_extended=include_extended,
        use_gpu=True,
    )

    return results_to_dict(results)


@app.local_entrypoint()
def main(
    datasets: str | None = None,
    configs: str | None = None,
    extended: bool = False,
):
    """Run benchmark on Modal.

    Args:
        datasets: Comma-separated dataset names (e.g., "cpu_act,higgs")
        configs: Comma-separated config names (e.g., "baseline,deep_tree")
        extended: Include extended datasets
    """
    dataset_list = datasets.split(",") if datasets else None
    config_list = configs.split(",") if configs else None

    print("Running OpenBoost integration tests on Modal A100...")
    print(f"Datasets: {dataset_list or 'primary'}")
    print(f"Configs: {config_list or 'default'}")
    print(f"Extended: {extended}")
    print("")

    results = benchmark_gpu.remote(
        datasets=dataset_list,
        configs=config_list,
        include_extended=extended,
    )

    # Save results
    results_dir = PROJECT_ROOT / "benchmarks" / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"integration_results_{timestamp}.json"

    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print(f"\nSummary: {results['summary']}")


# =============================================================================
# Local Execution
# =============================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenBoost Integration Tests")
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally on CPU",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Dataset names to run (default: primary)",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        help="Config names to run (default: baseline, deep_tree, regularized, shallow_fast)",
    )
    parser.add_argument(
        "--extended",
        action="store_true",
        help="Include extended datasets",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples per dataset (for quick testing)",
    )

    args = parser.parse_args()

    if args.local:
        print("Running locally on CPU...")
        import sys

        sys.path.insert(0, str(PROJECT_ROOT / "src"))

        # Use smaller dataset size for local testing
        max_samples = args.max_samples or 10_000

        results = run_integration_tests(
            datasets=args.datasets,
            configs=args.configs or ["baseline", "shallow_fast"],
            include_extended=args.extended,
            use_gpu=False,
            max_samples=max_samples,
        )

        # Save results
        results_dir = PROJECT_ROOT / "benchmarks" / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = results_dir / f"integration_results_local_{timestamp}.json"

        with open(results_file, "w") as f:
            json.dump(results_to_dict(results), f, indent=2)

        print(f"\nResults saved to: {results_file}")
    else:
        print("Usage:")
        print("  Modal:  uv run modal run benchmarks/openml_integration.py")
        print("  Local:  uv run python benchmarks/openml_integration.py --local")
        print("")
        print("Options:")
        print("  --datasets cpu_act higgs     Run specific datasets")
        print("  --configs baseline deep_tree Run specific configs")
        print("  --extended                   Include extended datasets")
        print("  --max-samples 5000           Limit samples (for testing)")
