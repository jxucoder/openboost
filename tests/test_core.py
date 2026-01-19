"""Core tests for OpenBoost.

These tests run on CPU (for Mac development) and verify basic functionality.
"""

import numpy as np
import pytest

import openboost as ob


class TestArray:
    """Tests for ob.array() and binning."""
    
    def test_basic_binning(self):
        """Test that array() bins data correctly."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        binned = ob.array(X, n_bins=256)
        
        assert binned.n_samples == 100
        assert binned.n_features == 5
        assert binned.data.shape == (5, 100)  # Feature-major
        assert binned.data.dtype == np.uint8
    
    def test_bin_range(self):
        """Test that bin values are in valid range."""
        X = np.random.randn(1000, 10)
        binned = ob.array(X)
        
        assert binned.data.min() >= 0
        assert binned.data.max() <= 255
    
    def test_bin_edges_stored(self):
        """Test that bin edges are stored for inverse transform."""
        X = np.random.randn(100, 3)
        binned = ob.array(X)
        
        assert len(binned.bin_edges) == 3
        for edges in binned.bin_edges:
            assert len(edges) > 0  # At least some bin edges
    
    def test_n_bins_capped_at_255(self):
        """Test that n_bins > 255 is capped (Phase 14: bin 255 reserved for NaN)."""
        X = np.random.randn(10, 2)
        
        # n_bins > 255 should be silently capped to 255
        # (bin 255 is reserved for missing values)
        binned = ob.array(X, n_bins=300)
        
        # Data should not contain bin 255 (no NaN in this data)
        assert np.max(binned.data) < 255
    
    def test_invalid_shape(self):
        """Test that 1D input raises error."""
        X = np.random.randn(100)
        
        with pytest.raises(ValueError, match="must be 2D"):
            ob.array(X)


class TestFitTree:
    """Tests for ob.fit_tree()."""
    
    def test_basic_fit(self):
        """Test basic tree fitting."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 0.5 * X[:, 1]  # Simple linear target
        
        binned = ob.array(X)
        
        # MSE gradients
        pred = np.zeros(100, dtype=np.float32)
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(100, dtype=np.float32) * 2
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        
        assert tree.n_nodes > 0
        assert tree.depth <= 3
    
    def test_tree_prediction(self):
        """Test that tree predictions work."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0]  # Simple target
        
        binned = ob.array(X)
        
        pred = np.zeros(100, dtype=np.float32)
        grad = (2 * (pred - y)).astype(np.float32)
        hess = np.ones(100, dtype=np.float32) * 2
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=4)
        predictions = tree(binned)
        
        assert predictions.shape == (100,)
        assert predictions.dtype == np.float32
    
    def test_training_reduces_loss(self):
        """Test that multiple rounds reduce loss."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2]
        y = y.astype(np.float32)
        
        binned = ob.array(X)
        pred = np.zeros(200, dtype=np.float32)
        
        initial_loss = np.mean((pred - y) ** 2)
        
        # Train for a few rounds
        for _ in range(10):
            grad = (2 * (pred - y)).astype(np.float32)
            hess = np.ones(200, dtype=np.float32) * 2
            tree = ob.fit_tree(binned, grad, hess, max_depth=4)
            pred = pred + 0.3 * tree(binned)
        
        final_loss = np.mean((pred - y) ** 2)
        
        assert final_loss < initial_loss, f"Loss should decrease: {final_loss} < {initial_loss}"
    
    def test_max_depth_respected(self):
        """Test that max_depth is respected."""
        X = np.random.randn(100, 5)
        binned = ob.array(X)
        
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        for depth in [1, 2, 3, 5]:
            tree = ob.fit_tree(binned, grad, hess, max_depth=depth)
            assert tree.depth <= depth, f"Tree depth {tree.depth} > max_depth {depth}"


class TestBackend:
    """Tests for backend detection and dispatch."""
    
    def test_get_backend(self):
        """Test that get_backend returns valid value."""
        backend = ob.get_backend()
        assert backend in ("cuda", "cpu")
    
    def test_set_backend_cpu(self):
        """Test forcing CPU backend."""
        original = ob.get_backend()
        try:
            ob.set_backend("cpu")
            assert ob.get_backend() == "cpu"
            assert ob.is_cpu()
            assert not ob.is_cuda()
        finally:
            # Restore
            if original == "cuda":
                try:
                    ob.set_backend("cuda")
                except RuntimeError:
                    pass  # CUDA not available
    
    def test_invalid_backend(self):
        """Test that invalid backend raises error."""
        with pytest.raises(ValueError, match="must be 'cuda' or 'cpu'"):
            ob.set_backend("invalid")


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_constant_feature(self):
        """Test handling of constant features."""
        X = np.random.randn(100, 3)
        X[:, 1] = 5.0  # Constant feature
        
        binned = ob.array(X)
        
        # Should still work
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        
        assert tree.n_nodes > 0
    
    def test_small_dataset(self):
        """Test with very small dataset."""
        X = np.random.randn(10, 2)
        binned = ob.array(X)
        
        grad = np.random.randn(10).astype(np.float32)
        hess = np.ones(10, dtype=np.float32)
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=2)
        pred = tree(binned)
        
        assert pred.shape == (10,)
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        binned = ob.array(X)
        
        grad = np.random.randn(100).astype(np.float32)
        hess = np.ones(100, dtype=np.float32)
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        pred = tree(binned)
        
        assert pred.shape == (100,)


class TestGradientBoosting:
    """Tests for high-level GradientBoosting API."""
    
    def test_import(self):
        """Test that openboost can be imported."""
        assert hasattr(ob, "GradientBoosting")
        assert ob.__version__ == "0.1.0"
    
    def test_fit_predict_small(self):
        """Test fit/predict on small data."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=3, max_depth=3)
        model.fit(X, y)
        
        pred = model.predict(X)
        
        assert pred.shape == y.shape
        assert pred.dtype == np.float32
    
    def test_mse_decreases(self):
        """Test that MSE decreases with more trees."""
        np.random.seed(42)
        X = np.random.randn(200, 10).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200).astype(np.float32) * 0.1
        
        model_few = ob.GradientBoosting(n_trees=2, max_depth=3)
        model_few.fit(X, y)
        mse_few = np.mean((model_few.predict(X) - y) ** 2)
        
        model_many = ob.GradientBoosting(n_trees=10, max_depth=3)
        model_many.fit(X, y)
        mse_many = np.mean((model_many.predict(X) - y) ** 2)
        
        assert mse_many < mse_few, f"More trees should reduce MSE: {mse_many} >= {mse_few}"
    
    def test_learning_rate_effect(self):
        """Test that lower learning rate requires more trees."""
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
    
    def test_deterministic(self):
        """Test that results are deterministic."""
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
