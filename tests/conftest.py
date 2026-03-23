"""Shared test fixtures for OpenBoost.

Centralizes dataset generation, pre-binned arrays, and gradient fixtures
to eliminate duplication across test files. All data fixtures use explicit
RandomState objects (not np.random.seed) to avoid cross-test contamination.
"""

import numpy as np
import pytest

import openboost as ob

# =============================================================================
# CUDA detection and auto-skip
# =============================================================================

try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False


def pytest_collection_modifyitems(config, items):
    """Auto-skip GPU and parity tests when CUDA is unavailable."""
    if not CUDA_AVAILABLE:
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords or "parity" in item.keywords:
                item.add_marker(skip_gpu)


# =============================================================================
# Regression datasets
# =============================================================================

@pytest.fixture(scope="session")
def regression_100x5():
    """Small regression dataset: 100 samples, 5 features, linear target.

    y = X[:,0] + 0.5 * X[:,1] + noise(0.1)
    """
    rng = np.random.RandomState(42)
    X = rng.randn(100, 5).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(100).astype(np.float32) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture(scope="session")
def regression_200x10():
    """Medium regression dataset: 200 samples, 10 features, linear target.

    y = X[:,0] + 0.5 * X[:,1] - 0.3 * X[:,2] + noise(0.1)
    """
    rng = np.random.RandomState(42)
    X = rng.randn(200, 10).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.randn(200).astype(np.float32) * 0.1).astype(np.float32)
    return X, y


@pytest.fixture(scope="session")
def regression_500x10():
    """Larger regression dataset: 500 samples, 10 features.

    y = X[:,0] + 0.5 * X[:,1] - 0.3 * X[:,2] + noise(0.1)
    """
    rng = np.random.RandomState(42)
    X = rng.randn(500, 10).astype(np.float32)
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.randn(500).astype(np.float32) * 0.1).astype(np.float32)
    return X, y


# =============================================================================
# Classification datasets
# =============================================================================

@pytest.fixture(scope="session")
def binary_500x10():
    """Binary classification dataset: 500 samples, 10 features.

    Labels derived from a linear boundary on first two features.
    """
    rng = np.random.RandomState(42)
    X = rng.randn(500, 10).astype(np.float32)
    logits = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
    y = (logits > 0).astype(np.float32)
    return X, y


@pytest.fixture(scope="session")
def multiclass_300x5():
    """3-class classification dataset: 300 samples, 5 features."""
    rng = np.random.RandomState(42)
    X = rng.randn(300, 5).astype(np.float32)
    # 3 classes based on which of 3 linear combos is largest
    scores = np.column_stack([X[:, 0], X[:, 1], X[:, 2]])
    y = scores.argmax(axis=1).astype(np.float32)
    return X, y


# =============================================================================
# Specialized datasets
# =============================================================================

@pytest.fixture(scope="session")
def count_data_200x5():
    """Poisson count data: 200 samples, 5 features.

    y ~ Poisson(exp(0.5 * X[:,0] + 0.3 * X[:,1]))
    """
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5).astype(np.float32)
    rate = np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1])
    y = rng.poisson(rate).astype(np.float32)
    return X, y


@pytest.fixture(scope="session")
def positive_continuous_200x5():
    """Positive continuous data for Gamma/Tweedie: 200 samples, 5 features.

    y = exp(0.5 * X[:,0] + 0.3 * X[:,1]) + noise
    """
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5).astype(np.float32)
    y = (np.exp(0.5 * X[:, 0] + 0.3 * X[:, 1]) + rng.exponential(0.1, 200)).astype(np.float32)
    return X, y


# =============================================================================
# Pre-binned datasets
# =============================================================================

@pytest.fixture(scope="session")
def binned_100x5(regression_100x5):
    """Pre-binned version of regression_100x5."""
    X, y = regression_100x5
    return ob.array(X), y


@pytest.fixture(scope="session")
def binned_200x10(regression_200x10):
    """Pre-binned version of regression_200x10."""
    X, y = regression_200x10
    return ob.array(X), y


# =============================================================================
# Gradient/hessian fixtures
# =============================================================================

@pytest.fixture
def mse_grads_100(regression_100x5):
    """MSE gradients from zero predictions for regression_100x5.

    Returns (grad, hess) with grad = 2*(0-y) = -2y, hess = 2.
    """
    _, y = regression_100x5
    pred = np.zeros(100, dtype=np.float32)
    grad = (2 * (pred - y)).astype(np.float32)
    hess = np.ones(100, dtype=np.float32) * 2
    return grad, hess


@pytest.fixture
def mse_grads_200(regression_200x10):
    """MSE gradients from zero predictions for regression_200x10."""
    _, y = regression_200x10
    pred = np.zeros(200, dtype=np.float32)
    grad = (2 * (pred - y)).astype(np.float32)
    hess = np.ones(200, dtype=np.float32) * 2
    return grad, hess


# =============================================================================
# Pre-fitted model fixtures
# =============================================================================

@pytest.fixture
def fitted_regressor(regression_500x10):
    """Pre-fitted OpenBoostRegressor (20 trees, max_depth=4).

    Function-scoped: fresh model for each test.
    """
    X, y = regression_500x10
    model = ob.GradientBoosting(n_trees=20, max_depth=4, learning_rate=0.1)
    model.fit(X, y)
    return model, X, y
