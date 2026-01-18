"""Tests for loss functions (Phase 9.1)."""

import numpy as np
import pytest

import openboost as ob


class TestMAELoss:
    """Tests for MAE (L1) loss."""
    
    def test_gradient_sign(self):
        """Test that MAE gradient is sign of residual."""
        pred = np.array([1.0, 2.0, 3.0, 2.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 0.0, 5.0], dtype=np.float32)
        
        grad, hess = ob.mae_gradient(pred, y)
        
        # grad = sign(pred - y)
        expected_grad = np.array([1.0, 1.0, 1.0, -1.0])
        np.testing.assert_array_almost_equal(grad, expected_grad)
        
        # hess = constant
        assert np.all(hess > 0)
    
    def test_gradient_zero_at_match(self):
        """Test gradient is zero when pred == y."""
        pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        y = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        grad, hess = ob.mae_gradient(pred, y)
        
        np.testing.assert_array_almost_equal(grad, [0.0, 0.0, 0.0])
    
    def test_with_gradient_boosting(self):
        """Test MAE loss with GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(200) * 0.1
        
        model = ob.GradientBoosting(n_trees=20, loss='mae')
        model.fit(X, y)
        
        pred = model.predict(X)
        mae = np.mean(np.abs(pred - y))
        
        # Should reduce error
        initial_mae = np.mean(np.abs(y))
        assert mae < initial_mae


class TestQuantileLoss:
    """Tests for Quantile (Pinball) loss."""
    
    def test_gradient_median(self):
        """Test gradient for median (alpha=0.5) is similar to MAE."""
        pred = np.array([1.0, 2.0, 0.0], dtype=np.float32)
        y = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        grad, hess = ob.quantile_gradient(pred, y, alpha=0.5)
        
        # At alpha=0.5: grad = 0.5 if pred > y (over), -0.5 if pred < y (under)
        expected_grad = np.array([0.5, 0.5, -0.5])
        np.testing.assert_array_almost_equal(grad, expected_grad)
    
    def test_gradient_high_quantile(self):
        """Test gradient for high quantile (alpha=0.9)."""
        pred = np.array([1.0, 0.0], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)
        
        grad, hess = ob.quantile_gradient(pred, y, alpha=0.9)
        
        # pred[0] > y[0] (over-prediction): grad = 1 - alpha = 0.1
        # pred[1] < y[1] (under-prediction): grad = -alpha = -0.9
        expected_grad = np.array([0.1, -0.9])
        np.testing.assert_array_almost_equal(grad, expected_grad)
    
    def test_gradient_low_quantile(self):
        """Test gradient for low quantile (alpha=0.1)."""
        pred = np.array([1.0, 0.0], dtype=np.float32)
        y = np.array([0.0, 1.0], dtype=np.float32)
        
        grad, hess = ob.quantile_gradient(pred, y, alpha=0.1)
        
        # pred[0] > y[0] (over-prediction): grad = 1 - alpha = 0.9
        # pred[1] < y[1] (under-prediction): grad = -alpha = -0.1
        expected_grad = np.array([0.9, -0.1])
        np.testing.assert_array_almost_equal(grad, expected_grad)
    
    def test_with_gradient_boosting(self):
        """Test quantile loss with GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + np.random.randn(200) * 0.5
        
        # High quantile should predict above median
        model_high = ob.GradientBoosting(n_trees=30, loss='quantile', quantile_alpha=0.9)
        model_high.fit(X, y)
        pred_high = model_high.predict(X)
        
        # Low quantile should predict below median  
        model_low = ob.GradientBoosting(n_trees=30, loss='quantile', quantile_alpha=0.1)
        model_low.fit(X, y)
        pred_low = model_low.predict(X)
        
        # Median
        model_med = ob.GradientBoosting(n_trees=30, loss='quantile', quantile_alpha=0.5)
        model_med.fit(X, y)
        pred_med = model_med.predict(X)
        
        # High quantile predictions should generally be higher than low
        assert np.mean(pred_high) > np.mean(pred_low)
        # Median should be between
        assert np.mean(pred_low) < np.mean(pred_med) < np.mean(pred_high)


class TestGetLossFunction:
    """Tests for get_loss_function."""
    
    def test_mae_aliases(self):
        """Test that mae aliases work."""
        for name in ['mae', 'l1', 'absolute_error']:
            loss_fn = ob.get_loss_function(name)
            grad, hess = loss_fn(np.array([1.0]), np.array([0.0]))
            assert grad[0] == pytest.approx(1.0)
    
    def test_quantile_with_alpha(self):
        """Test quantile loss with custom alpha."""
        loss_fn = ob.get_loss_function('quantile', quantile_alpha=0.9)
        
        pred = np.array([1.0], dtype=np.float32)
        y = np.array([0.0], dtype=np.float32)
        
        grad, hess = loss_fn(pred, y)
        
        # pred > y (over-prediction), so grad = 1 - alpha = 0.1
        assert grad[0] == pytest.approx(0.1)
    
    def test_unknown_loss_error(self):
        """Test error for unknown loss."""
        with pytest.raises(ValueError, match="Unknown loss"):
            ob.get_loss_function('unknown_loss')


class TestLossesWithFitTree:
    """Integration tests: losses with fit_tree."""
    
    def test_mae_fit_tree(self):
        """Test MAE gradient with fit_tree."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0] + 0.5 * X[:, 1]
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad, hess = ob.mae_gradient(pred, y)
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        
        assert tree.n_nodes > 0
        
        new_pred = tree.predict(binned.data)
        # Should make some improvement
        new_mae = np.mean(np.abs(new_pred - y))
        old_mae = np.mean(np.abs(y))
        assert new_mae < old_mae
    
    def test_quantile_fit_tree(self):
        """Test quantile gradient with fit_tree."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = X[:, 0]
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad, hess = ob.quantile_gradient(pred, y, alpha=0.5)
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        
        assert tree.n_nodes > 0
        assert not np.any(np.isnan(tree.predict(binned.data)))


class TestPoissonLoss:
    """Tests for Poisson loss (Phase 9.3)."""
    
    def test_gradient_basic(self):
        """Test Poisson gradient computation."""
        # pred in log-space, y is count
        pred = np.array([0.0, 1.0, 2.0], dtype=np.float32)  # exp = [1, 2.7, 7.4]
        y = np.array([1.0, 3.0, 7.0], dtype=np.float32)
        
        grad, hess = ob.poisson_gradient(pred, y)
        
        # grad = exp(pred) - y
        exp_pred = np.exp(pred)
        expected_grad = exp_pred - y
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=4)
        
        # hess = exp(pred)
        np.testing.assert_array_almost_equal(hess, exp_pred, decimal=4)
    
    def test_with_gradient_boosting(self):
        """Test Poisson loss with GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        # Generate count data
        rate = np.exp(X[:, 0] * 0.5)  # True rate
        y = np.random.poisson(rate).astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=20, loss='poisson')
        model.fit(X, y)
        
        pred = model.predict(X)
        # Predictions should be reasonable
        assert not np.any(np.isnan(pred))


class TestGammaLoss:
    """Tests for Gamma loss (Phase 9.3)."""
    
    def test_gradient_basic(self):
        """Test Gamma gradient computation."""
        pred = np.array([0.0, 1.0], dtype=np.float32)
        y = np.array([1.0, 2.0], dtype=np.float32)
        
        grad, hess = ob.gamma_gradient(pred, y)
        
        # grad = 1 - y * exp(-pred)
        exp_neg_pred = np.exp(-pred)
        expected_grad = 1.0 - y * exp_neg_pred
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=4)
    
    def test_with_gradient_boosting(self):
        """Test Gamma loss with GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        # Generate positive continuous data
        y = np.exp(X[:, 0] * 0.3 + 1) + 0.1  # Always positive
        y = y.astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=20, loss='gamma')
        model.fit(X, y)
        
        pred = model.predict(X)
        assert not np.any(np.isnan(pred))


class TestTweedieLoss:
    """Tests for Tweedie loss (Phase 9.3)."""
    
    def test_gradient_basic(self):
        """Test Tweedie gradient computation."""
        pred = np.array([0.0, 1.0], dtype=np.float32)
        y = np.array([0.0, 2.0], dtype=np.float32)  # Can have zeros
        rho = 1.5
        
        grad, hess = ob.tweedie_gradient(pred, y, rho=rho)
        
        mu = np.exp(pred)
        expected_grad = np.power(mu, 2 - rho) - y * np.power(mu, 1 - rho)
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=4)
    
    def test_different_rho_values(self):
        """Test Tweedie with different rho values."""
        pred = np.array([0.5], dtype=np.float32)
        y = np.array([1.0], dtype=np.float32)
        
        # rho closer to 1 (more Poisson-like)
        grad1, hess1 = ob.tweedie_gradient(pred, y, rho=1.1)
        
        # rho closer to 2 (more Gamma-like)
        grad2, hess2 = ob.tweedie_gradient(pred, y, rho=1.9)
        
        # Both should be valid
        assert not np.isnan(grad1[0])
        assert not np.isnan(grad2[0])
    
    def test_with_gradient_boosting(self):
        """Test Tweedie loss with GradientBoosting."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        # Generate data with zeros (insurance-like)
        y = np.maximum(X[:, 0] * 2 + np.random.randn(200), 0).astype(np.float32)
        
        model = ob.GradientBoosting(n_trees=20, loss='tweedie', tweedie_rho=1.5)
        model.fit(X, y)
        
        pred = model.predict(X)
        assert not np.any(np.isnan(pred))


class TestSoftmaxLoss:
    """Tests for Softmax loss (Phase 9.2)."""
    
    def test_gradient_shape(self):
        """Test softmax gradient shape."""
        n_samples, n_classes = 100, 5
        pred = np.random.randn(n_samples, n_classes).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)
        
        grad, hess = ob.softmax_gradient(pred, y, n_classes)
        
        assert grad.shape == (n_samples, n_classes)
        assert hess.shape == (n_samples, n_classes)
    
    def test_gradient_sums_to_zero(self):
        """Test that softmax gradient sums to ~0 per sample (prob sums to 1)."""
        n_samples, n_classes = 50, 3
        pred = np.random.randn(n_samples, n_classes).astype(np.float32)
        y = np.random.randint(0, n_classes, n_samples)
        
        grad, hess = ob.softmax_gradient(pred, y, n_classes)
        
        # For each sample, sum of gradients should be close to 0
        # (since sum of probs = 1 and one-hot sums to 1)
        grad_sums = np.sum(grad, axis=1)
        np.testing.assert_array_almost_equal(grad_sums, np.zeros(n_samples), decimal=5)
    
    def test_correct_class_gets_negative_gradient(self):
        """Test that correct class has negative gradient (reduce loss)."""
        n_classes = 3
        # Start with uniform predictions
        pred = np.zeros((1, n_classes), dtype=np.float32)
        y = np.array([1])  # Correct class is 1
        
        grad, hess = ob.softmax_gradient(pred, y, n_classes)
        
        # Gradient for correct class should be negative (prob - 1 < 0)
        # since prob = 1/3 initially
        assert grad[0, 1] < 0  # Correct class
        assert grad[0, 0] > 0  # Wrong class
        assert grad[0, 2] > 0  # Wrong class


class TestMultiClassGradientBoosting:
    """Tests for MultiClassGradientBoosting."""
    
    def test_basic_fit(self):
        """Test basic multi-class fitting."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = (X[:, 0] > 0).astype(np.int32) + (X[:, 1] > 0).astype(np.int32)  # 3 classes
        
        model = ob.MultiClassGradientBoosting(n_classes=3, n_trees=10)
        model.fit(X, y)
        
        assert len(model.trees_) == 10
        assert len(model.trees_[0]) == 3  # K trees per round
    
    def test_predict_proba_shape(self):
        """Test predict_proba output shape."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 4, 100)
        
        model = ob.MultiClassGradientBoosting(n_classes=4, n_trees=5)
        model.fit(X, y)
        
        proba = model.predict_proba(X)
        
        assert proba.shape == (100, 4)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(100))
    
    def test_predict_returns_labels(self):
        """Test predict returns integer labels."""
        np.random.seed(42)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randint(0, 3, 100)
        
        model = ob.MultiClassGradientBoosting(n_classes=3, n_trees=5)
        model.fit(X, y)
        
        pred = model.predict(X)
        
        assert pred.shape == (100,)
        assert pred.dtype in (np.int32, np.int64)
        assert np.all((pred >= 0) & (pred < 3))
    
    def test_training_improves_accuracy(self):
        """Test that training improves accuracy."""
        np.random.seed(42)
        # Create separable data
        X = np.random.randn(300, 5).astype(np.float32)
        y = (X[:, 0] + X[:, 1] > 0).astype(np.int32) + (X[:, 2] > 0.5).astype(np.int32)
        
        model = ob.MultiClassGradientBoosting(n_classes=3, n_trees=30)
        model.fit(X, y)
        
        pred = model.predict(X)
        accuracy = np.mean(pred == y)
        
        # Should do better than random (33%)
        assert accuracy > 0.5
    
    def test_invalid_labels_error(self):
        """Test error on invalid labels."""
        X = np.random.randn(50, 3).astype(np.float32)
        y = np.array([0, 1, 5])  # 5 is invalid for n_classes=3
        
        model = ob.MultiClassGradientBoosting(n_classes=3, n_trees=5)
        
        with pytest.raises(ValueError, match="Labels must be"):
            model.fit(X, y)
    
    def test_not_fitted_error(self):
        """Test error when predicting before fitting."""
        model = ob.MultiClassGradientBoosting(n_classes=3)
        
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.random.randn(10, 5))


class TestGLMLossesWithFitTree:
    """Integration tests: GLM losses with fit_tree."""
    
    def test_poisson_fit_tree(self):
        """Test Poisson gradient with fit_tree."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = np.abs(X[:, 0] * 2).astype(np.float32) + 1  # Positive
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad, hess = ob.poisson_gradient(pred, y)
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        
        assert tree.n_nodes > 0
        assert not np.any(np.isnan(tree.predict(binned.data)))
    
    def test_gamma_fit_tree(self):
        """Test Gamma gradient with fit_tree."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = np.exp(X[:, 0] * 0.5) + 0.1  # Positive
        y = y.astype(np.float32)
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad, hess = ob.gamma_gradient(pred, y)
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        
        assert tree.n_nodes > 0
    
    def test_tweedie_fit_tree(self):
        """Test Tweedie gradient with fit_tree."""
        np.random.seed(42)
        X = np.random.randn(200, 5).astype(np.float32)
        y = np.maximum(X[:, 0] + np.random.randn(200) * 0.5, 0).astype(np.float32)
        
        binned = ob.array(X)
        
        pred = np.zeros(200, dtype=np.float32)
        grad, hess = ob.tweedie_gradient(pred, y, rho=1.5)
        
        tree = ob.fit_tree(binned, grad, hess, max_depth=3)
        
        assert tree.n_nodes > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
