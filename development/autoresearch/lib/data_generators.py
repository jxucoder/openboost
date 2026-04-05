"""Consolidated data generation for autoresearch v2 benchmarks.

All generators produce deterministic outputs from fixed seeds.
No network downloads — everything is synthetic.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Shared: mixed-feature matrix builder
# ---------------------------------------------------------------------------

def _make_features(
    n_samples: int, n_features: int, rng: np.random.RandomState
) -> tuple[np.ndarray, int, int]:
    """Build a mixed-feature matrix (float32).

    Feature distribution (% of n_features):
      20% gaussian, 20% uniform, 10% log-normal, 10% binary,
      10% ordinal, 10% correlated, rest noise.

    Returns (X, gauss_end_idx, binary_start_idx).
    """
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
        X[:, idx + j] = (0.8 * X[:, src] + 0.2 * rng.randn(n_samples)).astype(np.float32)
    idx += n_corr

    X[:, idx:idx + n_noise] = rng.randn(n_samples, n_noise)

    return X, gauss_end, binary_start


def _regression_target(
    X: np.ndarray, gauss_end: int, binary_start: int, rng: np.random.RandomState
) -> np.ndarray:
    """Non-linear target with interactions, thresholds, and noise."""
    n_samples = X.shape[0]
    return (
        2.0 * np.sin(X[:, 0] * 1.5)
        + 1.5 * X[:, 1] * X[:, 2]
        + np.where(X[:, 3] > 0, X[:, 4], -X[:, 4])
        + 0.8 * X[:, gauss_end] ** 2
        + 1.0 * X[:, binary_start]
        + 0.3 * X[:, 0] * X[:, binary_start]
        + rng.randn(n_samples).astype(np.float32) * 0.5
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Speed benchmark data
# ---------------------------------------------------------------------------

SPEED_CONFIGS = {
    "small_reg":  {"n_samples": 500_000,   "n_features": 50,  "task": "regression"},
    "small_bin":  {"n_samples": 500_000,   "n_features": 50,  "task": "binary"},
    "medium_reg": {"n_samples": 1_000_000, "n_features": 100, "task": "regression"},
    "medium_bin": {"n_samples": 1_000_000, "n_features": 100, "task": "binary"},
    "large_reg":  {"n_samples": 2_000_000, "n_features": 80,  "task": "regression"},
}


def make_speed_data(
    config_name: str, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, str]:
    """Generate data for a speed benchmark config.

    Returns (X, y, task).
    """
    cfg = SPEED_CONFIGS[config_name]
    rng = np.random.RandomState(seed)
    X, gauss_end, binary_start = _make_features(cfg["n_samples"], cfg["n_features"], rng)
    task = cfg["task"]

    if task == "regression":
        y = _regression_target(X, gauss_end, binary_start, rng)
    elif task == "binary":
        y_latent = _regression_target(X, gauss_end, binary_start, rng)
        threshold = np.percentile(y_latent, 70)
        prob = 1.0 / (1.0 + np.exp(-(y_latent - threshold)))
        y = (rng.rand(cfg["n_samples"]) < prob).astype(np.float32)
    else:
        raise ValueError(f"Unknown task: {task}")

    return X, y, task


# ---------------------------------------------------------------------------
# Accuracy benchmark data
# ---------------------------------------------------------------------------

def make_accuracy_regression(
    n_samples: int = 200_000, n_features: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Realistic regression dataset with train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.RandomState(seed)
    X, gauss_end, binary_start = _make_features(n_samples, n_features, rng)
    y = _regression_target(X, gauss_end, binary_start, rng)

    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def make_accuracy_binary(
    n_samples: int = 200_000, n_features: int = 50, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Realistic binary classification with train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.RandomState(seed)
    X, gauss_end, binary_start = _make_features(n_samples, n_features, rng)
    y_latent = _regression_target(X, gauss_end, binary_start, rng)
    threshold = np.percentile(y_latent, 70)
    prob = 1.0 / (1.0 + np.exp(-(y_latent - threshold)))
    y = (rng.rand(n_samples) < prob).astype(np.float32)

    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def make_accuracy_multiclass(
    n_samples: int = 200_000, n_features: int = 50, n_classes: int = 5, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Realistic multiclass classification with train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.RandomState(seed)
    X, _, _ = _make_features(n_samples, n_features, rng)

    # Class logits from different feature subsets with non-linearity
    logits = np.zeros((n_samples, n_classes), dtype=np.float32)
    features_per_class = max(1, n_features // n_classes)
    for c in range(n_classes):
        start = (c * features_per_class) % n_features
        end = min(start + features_per_class, n_features)
        weights = rng.randn(end - start).astype(np.float32)
        logits[:, c] = X[:, start:end] @ weights + 0.3 * np.sin(X[:, c % n_features] * 2)

    # Softmax + sampling
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(n_classes, p=p) for p in probs], dtype=np.int32)

    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def make_accuracy_poisson(
    n_samples: int = 50_000, n_features: int = 20, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Poisson count data with train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    log_mu = 1.0 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2]
    mu = np.exp(np.clip(log_mu, -5, 5))
    y = rng.poisson(mu).astype(np.float32)

    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


# ---------------------------------------------------------------------------
# Coverage benchmark data
# ---------------------------------------------------------------------------

def make_missing_data(
    n_samples: int = 200_000,
    n_features: int = 50,
    missing_rate: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Regression data with injected NaN values. Train/test split.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.RandomState(seed)
    X, gauss_end, binary_start = _make_features(n_samples, n_features, rng)
    y = _regression_target(X, gauss_end, binary_start, rng)

    # Inject NaN
    mask = rng.rand(n_samples, n_features) < missing_rate
    X[mask] = np.nan

    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]


def make_categorical_data(
    n_samples: int = 50_000,
    n_numeric: int = 10,
    n_categorical: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    """Regression data with categorical features. Train/test split.

    Returns (X_train, X_test, y_train, y_test, categorical_indices).
    """
    rng = np.random.RandomState(seed)
    n_features = n_numeric + n_categorical

    X = np.empty((n_samples, n_features), dtype=np.float32)
    # Numeric features
    X[:, :n_numeric] = rng.randn(n_samples, n_numeric).astype(np.float32)
    # Categorical features (integer-valued, 3-10 categories each)
    cat_indices = list(range(n_numeric, n_features))
    for i, col in enumerate(cat_indices):
        n_cats = 3 + (i * 7) % 8  # 3 to 10 categories
        X[:, col] = rng.randint(0, n_cats, n_samples).astype(np.float32)

    # Target depends on both numeric and categorical
    y = (
        X[:, 0] * 2.0
        + np.sin(X[:, 1] * 1.5)
        + X[:, n_numeric] * 0.5  # first categorical
        + np.where(X[:, n_numeric + 1] > 2, 1.0, -1.0)  # second categorical threshold
        + rng.randn(n_samples).astype(np.float32) * 0.3
    ).astype(np.float32)

    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:], cat_indices


def make_small_regression(
    n_samples: int = 10_000, n_features: int = 20, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Small regression dataset for variant model coverage tests.

    Returns (X_train, X_test, y_train, y_test).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (
        np.sin(X[:, 0] * 2)
        + 0.5 * X[:, 1] ** 2
        + 0.3 * X[:, 2] * X[:, 3]
        + rng.randn(n_samples).astype(np.float32) * 0.1
    ).astype(np.float32)

    split = int(n_samples * 0.8)
    return X[:split], X[split:], y[:split], y[split:]
