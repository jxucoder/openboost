"""Core tests - run with NUMBA_ENABLE_CUDASIM=1 for CPU simulation."""

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def setup_cuda_sim(monkeypatch):
    """Enable CUDA simulator for all tests."""
    monkeypatch.setenv("NUMBA_ENABLE_CUDASIM", "1")


def test_import():
    """Test that openboost can be imported."""
    import openboost as ob
    assert hasattr(ob, "GradientBoosting")
    assert ob.__version__ == "0.1.0"


def test_fit_predict_small():
    """Test fit/predict on small data."""
    import openboost as ob
    
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    model = ob.GradientBoosting(n_trees=3, max_depth=3)
    model.fit(X, y)
    
    pred = model.predict(X)
    
    assert pred.shape == y.shape
    assert pred.dtype == np.float32


def test_mse_decreases():
    """Test that MSE decreases with more trees."""
    import openboost as ob
    
    np.random.seed(42)
    X = np.random.randn(200, 10).astype(np.float32)
    y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200).astype(np.float32) * 0.1
    
    model_few = ob.GradientBoosting(n_trees=2, max_depth=3)
    model_few.fit(X, y)
    mse_few = np.mean((model_few.predict(X) - y) ** 2)
    
    model_many = ob.GradientBoosting(n_trees=10, max_depth=3)
    model_many.fit(X, y)
    mse_many = np.mean((model_many.predict(X) - y) ** 2)
    
    # temporary for now because it may not be true for all datasets
    assert mse_many < mse_few, f"More trees should reduce MSE: {mse_many} >= {mse_few}"


def test_learning_rate_effect():
    """Test that lower learning rate requires more trees."""
    import openboost as ob
    
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = X[:, 0] * 2 + np.random.randn(100).astype(np.float32) * 0.1
    
    # High LR, few trees
    model_high_lr = ob.GradientBoosting(n_trees=5, max_depth=3, learning_rate=0.5)
    model_high_lr.fit(X, y)
    pred_high = model_high_lr.predict(X)
    
    # Low LR, few trees - should fit worse
    model_low_lr = ob.GradientBoosting(n_trees=5, max_depth=3, learning_rate=0.01)
    model_low_lr.fit(X, y)
    pred_low = model_low_lr.predict(X)
    
    mse_high = np.mean((pred_high - y) ** 2)
    mse_low = np.mean((pred_low - y) ** 2)
    
    # With same trees, high LR should fit better (for small n_trees)
    assert mse_high < mse_low


def test_deterministic():
    """Test that results are deterministic."""
    import openboost as ob
    
    np.random.seed(42)
    X = np.random.randn(50, 3).astype(np.float32)
    y = np.random.randn(50).astype(np.float32)
    
    model1 = ob.GradientBoosting(n_trees=3, max_depth=2)
    model1.fit(X, y)
    pred1 = model1.predict(X)
    
    model2 = ob.GradientBoosting(n_trees=3, max_depth=2)
    model2.fit(X, y)
    pred2 = model2.predict(X)
    
    np.testing.assert_array_equal(pred1, pred2)

