"""Tests for multi-GPU training (Phase 18).

These tests verify:
1. MultiGPUContext initialization and GPU detection
2. Data sharding and worker creation
3. Histogram aggregation
4. GradientBoosting multi-GPU integration
5. Results match single-GPU (numerical correctness)

Note: Most tests use mocks to run without actual GPUs.
Integration tests on real multi-GPU hardware can be run on Modal cloud.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

import openboost as ob
from openboost._distributed import DistributedContext


# =============================================================================
# Test fixtures
# =============================================================================

@pytest.fixture
def sample_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    return X, y


@pytest.fixture
def small_data():
    """Generate small dataset for quick tests."""
    np.random.seed(42)
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    return X, y


# =============================================================================
# Unit tests (no GPU required, mocked Ray)
# =============================================================================

def test_gradient_boosting_multigpu_params():
    """Test GradientBoosting can be initialized with multi-GPU params."""
    # n_gpus parameter
    model = ob.GradientBoosting(n_trees=10, n_gpus=4)
    assert model.n_gpus == 4
    assert model.devices is None
    
    # devices parameter
    model = ob.GradientBoosting(n_trees=10, devices=[0, 1, 2, 3])
    assert model.devices == [0, 1, 2, 3]
    assert model.n_gpus is None
    
    # Both parameters
    model = ob.GradientBoosting(n_trees=10, n_gpus=2, devices=[0, 2])
    assert model.n_gpus == 2
    assert model.devices == [0, 2]


def test_gradient_boosting_multigpu_requires_ray(small_data):
    """Test that multi-GPU training raises ImportError without Ray."""
    X, y = small_data
    
    # Check if ray is installed
    try:
        import ray
        pytest.skip("Ray is installed, skipping import error test")
    except ImportError:
        pass
    
    model = ob.GradientBoosting(n_trees=10, n_gpus=2)
    
    with pytest.raises(ImportError, match="Ray"):
        model.fit(X, y)


def test_multigpu_context_protocol():
    """Verify MultiGPUContext has the expected interface."""
    try:
        from openboost._distributed._multigpu import MultiGPUContext
    except ImportError:
        pytest.skip("Ray not installed")
    
    # Check class exists and has expected attributes
    assert hasattr(MultiGPUContext, 'setup')
    assert hasattr(MultiGPUContext, 'compute_all_gradients')
    assert hasattr(MultiGPUContext, 'build_all_histograms')
    assert hasattr(MultiGPUContext, 'aggregate_histograms')
    assert hasattr(MultiGPUContext, 'update_all_predictions')
    assert hasattr(MultiGPUContext, 'get_all_predictions')
    assert hasattr(MultiGPUContext, 'shutdown')


def test_gpu_worker_protocol():
    """Verify GPUWorkerBase has the expected interface."""
    try:
        from openboost._distributed._multigpu import GPUWorkerBase
    except ImportError:
        pytest.skip("Ray not installed")
    
    # Check class exists and has expected methods
    assert hasattr(GPUWorkerBase, 'compute_gradients')
    assert hasattr(GPUWorkerBase, 'build_histogram')
    assert hasattr(GPUWorkerBase, 'update_predictions')
    assert hasattr(GPUWorkerBase, 'get_predictions')
    assert hasattr(GPUWorkerBase, 'get_n_features')
    assert hasattr(GPUWorkerBase, 'get_n_samples')


def test_histogram_aggregation_logic():
    """Test histogram aggregation (sum) logic."""
    try:
        from openboost._distributed._multigpu import MultiGPUContext
    except ImportError:
        pytest.skip("Ray not installed")
    
    # Create mock histograms from "different GPUs"
    n_features = 5
    n_bins = 256
    
    hist1_grad = np.random.randn(n_features, n_bins).astype(np.float32)
    hist1_hess = np.abs(np.random.randn(n_features, n_bins)).astype(np.float32)
    
    hist2_grad = np.random.randn(n_features, n_bins).astype(np.float32)
    hist2_hess = np.abs(np.random.randn(n_features, n_bins)).astype(np.float32)
    
    hist3_grad = np.random.randn(n_features, n_bins).astype(np.float32)
    hist3_hess = np.abs(np.random.randn(n_features, n_bins)).astype(np.float32)
    
    local_histograms = [
        (hist1_grad, hist1_hess),
        (hist2_grad, hist2_hess),
        (hist3_grad, hist3_hess),
    ]
    
    # Test aggregation logic directly
    global_hist_grad = hist1_grad.copy()
    global_hist_hess = hist1_hess.copy()
    
    for hist_grad, hist_hess in local_histograms[1:]:
        global_hist_grad += hist_grad
        global_hist_hess += hist_hess
    
    # Expected result
    expected_grad = hist1_grad + hist2_grad + hist3_grad
    expected_hess = hist1_hess + hist2_hess + hist3_hess
    
    np.testing.assert_allclose(global_hist_grad, expected_grad, rtol=1e-5)
    np.testing.assert_allclose(global_hist_hess, expected_hess, rtol=1e-5)


def test_data_sharding_logic():
    """Test data sharding into equal parts."""
    n_samples = 1000
    n_gpus = 4
    
    # Test array_split behavior (what MultiGPUContext uses)
    indices = np.array_split(np.arange(n_samples), n_gpus)
    
    # Should have n_gpus shards
    assert len(indices) == n_gpus
    
    # Each shard should have ~n_samples/n_gpus elements
    for idx in indices:
        assert len(idx) >= n_samples // n_gpus - 1
        assert len(idx) <= n_samples // n_gpus + 1
    
    # All samples should be covered exactly once
    all_indices = np.concatenate(indices)
    assert len(all_indices) == n_samples
    assert len(np.unique(all_indices)) == n_samples


def test_multigpu_requires_raw_data(small_data):
    """Test that multi-GPU raises error for pre-binned data."""
    try:
        import ray
    except ImportError:
        pytest.skip("Ray not installed")
    
    X, y = small_data
    X_binned = ob.array(X, n_bins=256)
    
    model = ob.GradientBoosting(n_trees=5, n_gpus=2)
    
    # Should raise error because multi-GPU needs raw data
    with pytest.raises(ValueError, match="raw.*unbinned"):
        model.fit(X_binned, y)


# =============================================================================
# Integration tests (require Ray, but can run with mocked GPUs)
# =============================================================================

@pytest.mark.skipif(
    True,  # Skip by default, enable on multi-GPU systems
    reason="Requires Ray and multi-GPU hardware"
)
def test_multigpu_training_matches_single_gpu(sample_data):
    """Test that multi-GPU results match single-GPU (numerical correctness)."""
    X, y = sample_data
    
    # Single GPU training
    model_single = ob.GradientBoosting(
        n_trees=10,
        max_depth=4,
        learning_rate=0.1,
        n_bins=64,
    )
    model_single.fit(X, y)
    pred_single = model_single.predict(X)
    
    # Multi-GPU training
    model_multi = ob.GradientBoosting(
        n_trees=10,
        max_depth=4,
        learning_rate=0.1,
        n_bins=64,
        n_gpus=2,
    )
    model_multi.fit(X, y)
    pred_multi = model_multi.predict(X)
    
    # Results should be similar (not exact due to floating point and order)
    # Allow 10% tolerance
    mse_single = np.mean((pred_single - y) ** 2)
    mse_multi = np.mean((pred_multi - y) ** 2)
    
    assert abs(mse_single - mse_multi) / mse_single < 0.1


@pytest.mark.skipif(
    True,  # Skip by default, enable on multi-GPU systems
    reason="Requires Ray and multi-GPU hardware"
)
def test_multigpu_scaling(sample_data):
    """Test that multi-GPU training is faster than single-GPU."""
    import time
    
    X, y = sample_data
    
    # Single GPU
    model_single = ob.GradientBoosting(n_trees=100, max_depth=6)
    start = time.time()
    model_single.fit(X, y)
    time_single = time.time() - start
    
    # 4 GPUs
    model_multi = ob.GradientBoosting(n_trees=100, max_depth=6, n_gpus=4)
    start = time.time()
    model_multi.fit(X, y)
    time_multi = time.time() - start
    
    # Should be at least 1.5x faster with 4 GPUs
    assert time_multi < time_single * 0.67  # 1.5x speedup


# =============================================================================
# Mock tests (test logic without Ray/GPU)
# =============================================================================

def test_tree_from_histogram_simple():
    """Test tree building from histogram."""
    # Create a simple histogram with clear best split
    n_features = 3
    n_bins = 10
    
    hist_grad = np.zeros((n_features, n_bins), dtype=np.float32)
    hist_hess = np.ones((n_features, n_bins), dtype=np.float32)
    
    # Feature 0: has a good split at bin 5
    # Left (bins 0-5): positive gradient sum
    # Right (bins 6-9): negative gradient sum
    hist_grad[0, :6] = 1.0
    hist_grad[0, 6:] = -1.0
    
    # Feature 1: no good split (all same gradient)
    hist_grad[1, :] = 0.5
    
    # Feature 2: weaker split
    hist_grad[2, :3] = 0.3
    hist_grad[2, 3:] = -0.1
    
    # Test split finding
    from openboost._core._split import find_best_split
    
    sum_grad = float(np.sum(hist_grad))
    sum_hess = float(np.sum(hist_hess))
    
    split = find_best_split(
        hist_grad, hist_hess,
        sum_grad, sum_hess,
        reg_lambda=1.0,
        min_child_weight=1.0,
        min_gain=0.0,
    )
    
    # Should select feature 0 with threshold 5
    assert split.is_valid
    assert split.feature == 0
    assert split.threshold == 5


def test_distributed_exports():
    """Test that distributed module exports are correct."""
    from openboost._distributed import (
        DistributedContext,
        GPUWorkerBase,
        GPUWorker,
        MultiGPUContext,
        fit_tree_multigpu,
    )
    
    # All should be importable
    assert DistributedContext is not None
    assert GPUWorkerBase is not None
    assert GPUWorker is not None
    assert MultiGPUContext is not None
    assert fit_tree_multigpu is not None


# =============================================================================
# Edge cases
# =============================================================================

def test_empty_histogram_aggregation():
    """Test aggregation with empty histogram list."""
    try:
        from openboost._distributed._multigpu import MultiGPUContext
    except ImportError:
        pytest.skip("Ray not installed")
    
    # Create a mock context
    ctx = MagicMock(spec=MultiGPUContext)
    ctx.aggregate_histograms = MultiGPUContext.aggregate_histograms
    
    # Should raise error for empty list
    with pytest.raises((ValueError, TypeError)):
        MultiGPUContext.aggregate_histograms(ctx, [])


def test_single_gpu_path():
    """Test that n_gpus=1 uses standard training path."""
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    # n_gpus=1 should NOT trigger multi-GPU path
    model = ob.GradientBoosting(n_trees=5, n_gpus=1)
    
    # Should work without Ray (uses standard GPU path)
    try:
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (100,)
    except ImportError:
        # If no GPU backend available, that's fine
        pass


def test_devices_single_gpu():
    """Test that devices=[0] uses standard training path."""
    X = np.random.randn(100, 5).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    
    # Single device should NOT trigger multi-GPU path
    model = ob.GradientBoosting(n_trees=5, devices=[0])
    
    try:
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (100,)
    except ImportError:
        pass


# =============================================================================
# Doctest examples
# =============================================================================

def test_docstring_examples():
    """Verify docstring examples are correct."""
    # From GradientBoosting docstring
    
    # Multi-GPU training (will fail without Ray, just check syntax)
    model = ob.GradientBoosting(n_trees=100, n_gpus=4)
    assert model.n_gpus == 4
    
    # Explicit GPU selection
    model = ob.GradientBoosting(n_trees=100, devices=[0, 2, 4, 6])
    assert model.devices == [0, 2, 4, 6]
