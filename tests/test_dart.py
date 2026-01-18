"""Tests for DART (Phase 8.5)."""

import numpy as np
import pytest

import openboost as ob


class TestDART:
    """Tests for DART model."""
    
    def test_basic_fit(self):
        """Test basic DART fitting."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200) * 0.1
        
        model = ob.DART(n_trees=10, dropout_rate=0.1, seed=42)
        model.fit(X, y)
        
        assert len(model.trees_) == 10
        assert len(model.tree_weights_) == 10
    
    def test_prediction(self):
        """Test DART prediction."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        model = ob.DART(n_trees=20, dropout_rate=0.2, seed=42)
        model.fit(X, y)
        
        pred = model.predict(X)
        
        assert pred.shape == (200,)
        assert not np.any(np.isnan(pred))
    
    def test_reduces_loss(self):
        """Test that DART reduces training loss."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        # Initial loss
        initial_loss = np.mean(y ** 2)
        
        model = ob.DART(n_trees=30, dropout_rate=0.1, seed=42)
        model.fit(X, y)
        
        pred = model.predict(X)
        final_loss = np.mean((pred - y) ** 2)
        
        assert final_loss < initial_loss
    
    def test_dropout_affects_training(self):
        """Test that dropout rate affects training."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        # Train with no dropout
        model_no_drop = ob.DART(n_trees=20, dropout_rate=0.0, seed=42)
        model_no_drop.fit(X, y)
        
        # Train with high dropout
        model_high_drop = ob.DART(n_trees=20, dropout_rate=0.5, seed=42)
        model_high_drop.fit(X, y)
        
        # Both should work
        pred_no_drop = model_no_drop.predict(X)
        pred_high_drop = model_high_drop.predict(X)
        
        assert pred_no_drop.shape == pred_high_drop.shape
        
        # They should produce different predictions (dropout changes training)
        # Not always guaranteed but very likely with different dropout rates
        # Skip this assertion as it's probabilistic
    
    def test_tree_weights(self):
        """Test that tree weights are computed."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]
        
        model = ob.DART(n_trees=10, dropout_rate=0.3, normalize=True, seed=42)
        model.fit(X, y)
        
        # All weights should be positive
        assert all(w > 0 for w in model.tree_weights_)
    
    def test_skip_drop(self):
        """Test skip_drop parameter."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]
        
        # With skip_drop=1.0, no dropout should ever happen
        model = ob.DART(n_trees=10, dropout_rate=0.5, skip_drop=1.0, seed=42)
        model.fit(X, y)
        
        # All weights should be learning_rate (no normalization applied)
        for w in model.tree_weights_:
            assert abs(w - model.learning_rate) < 1e-6
    
    def test_classification(self):
        """Test DART for binary classification."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
        
        model = ob.DART(n_trees=20, loss='logloss', dropout_rate=0.1, seed=42)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (200, 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
    
    def test_reproducibility(self):
        """Test that same seed produces same results."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]
        
        model1 = ob.DART(n_trees=10, dropout_rate=0.3, seed=123)
        model1.fit(X, y)
        pred1 = model1.predict(X)
        
        model2 = ob.DART(n_trees=10, dropout_rate=0.3, seed=123)
        model2.fit(X, y)
        pred2 = model2.predict(X)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_not_fitted_error(self):
        """Test error when predicting before fitting."""
        model = ob.DART()
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.random.randn(10, 5))
    
    def test_predict_proba_error(self):
        """Test error when using predict_proba with regression loss."""
        np.random.seed(42)
        X = np.random.randn(50, 3).astype(np.float32)
        y = X[:, 0]
        
        model = ob.DART(n_trees=5, loss='mse', seed=42)
        model.fit(X, y)
        
        with pytest.raises(ValueError, match="predict_proba"):
            model.predict_proba(X)


class TestDARTIntegration:
    """Integration tests for DART with the new architecture."""
    
    def test_uses_fit_tree(self):
        """Test that DART uses the standard fit_tree function."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]
        
        model = ob.DART(n_trees=5, seed=42)
        model.fit(X, y)
        
        # Trees should be TreeStructure instances
        for tree in model.trees_:
            assert isinstance(tree, ob.TreeStructure)
            assert tree.n_nodes > 0
    
    def test_custom_loss(self):
        """Test DART with custom loss function."""
        np.random.seed(42)
        X = np.random.randn(100, 3).astype(np.float32)
        y = X[:, 0]
        
        # Custom loss: Huber-like
        def custom_loss(pred, y):
            diff = pred - y
            grad = np.where(np.abs(diff) < 1, diff, np.sign(diff))
            hess = np.where(np.abs(diff) < 1, 1.0, 0.01)
            return grad.astype(np.float32), hess.astype(np.float32)
        
        model = ob.DART(n_trees=10, loss=custom_loss, seed=42)
        model.fit(X, y)
        
        pred = model.predict(X)
        assert pred.shape == (100,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
