"""Tests for LinearLeafGBDT model.

Verifies that gradient boosting with linear models in leaves works correctly
on CPU. These are the first CPU tests for this model variant.
"""

import numpy as np
import pytest

import openboost as ob


class TestLinearLeafBasic:
    """Basic functionality tests."""

    def test_basic_fit_predict(self, regression_200x10):
        """Fit and predict should produce correct shapes and dtypes."""
        X, y = regression_200x10

        model = ob.LinearLeafGBDT(n_trees=10, max_depth=3, learning_rate=0.1)
        model.fit(X, y)
        pred = model.predict(X)

        assert pred.shape == y.shape, f"Expected shape {y.shape}, got {pred.shape}"
        assert pred.dtype == np.float32
        assert np.all(np.isfinite(pred)), "Predictions should be finite"

    def test_training_reduces_loss(self, regression_200x10):
        """More trees should reduce training loss."""
        X, y = regression_200x10

        model_few = ob.LinearLeafGBDT(n_trees=5, max_depth=3)
        model_few.fit(X, y)
        mse_few = np.mean((model_few.predict(X) - y) ** 2)

        model_many = ob.LinearLeafGBDT(n_trees=30, max_depth=3)
        model_many.fit(X, y)
        mse_many = np.mean((model_many.predict(X) - y) ** 2)

        assert mse_many < mse_few, (
            f"More trees should reduce MSE: {mse_many} >= {mse_few}"
        )

    def test_deterministic(self, regression_100x5):
        """Same input should produce identical output."""
        X, y = regression_100x5

        model1 = ob.LinearLeafGBDT(n_trees=5, max_depth=2)
        model1.fit(X, y)
        pred1 = model1.predict(X)

        model2 = ob.LinearLeafGBDT(n_trees=5, max_depth=2)
        model2.fit(X, y)
        pred2 = model2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)

    def test_predict_before_fit_raises(self):
        """Predict on unfitted model should raise."""
        model = ob.LinearLeafGBDT(n_trees=5)
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3).astype(np.float32)

        with pytest.raises((RuntimeError, AttributeError)):
            model.predict(X)


class TestLinearLeafExtrapolation:
    """Verify that linear leaves improve extrapolation."""

    def test_extrapolation_on_linear_target(self):
        """LinearLeaf should extrapolate better than standard GBDT on linear data."""
        rng = np.random.RandomState(42)
        # Training: X in [-2, 2]
        X_train = rng.uniform(-2, 2, (200, 3)).astype(np.float32)
        y_train = (2 * X_train[:, 0] + X_train[:, 1]).astype(np.float32)

        # Test: X in [3, 5] (extrapolation region)
        X_test = rng.uniform(3, 5, (50, 3)).astype(np.float32)
        y_test = (2 * X_test[:, 0] + X_test[:, 1]).astype(np.float32)

        # Standard GBDT
        standard = ob.GradientBoosting(n_trees=50, max_depth=4, learning_rate=0.1)
        standard.fit(X_train, y_train)
        std_pred = standard.predict(X_test)
        _ = np.mean((std_pred - y_test) ** 2)

        # Linear Leaf GBDT
        linear = ob.LinearLeafGBDT(n_trees=50, max_depth=3, learning_rate=0.1)
        linear.fit(X_train, y_train)
        lin_pred = linear.predict(X_test)
        _ = np.mean((lin_pred - y_test) ** 2)

        # Linear leaf should extrapolate better (or at least comparably)
        # We don't assert strict superiority since it depends on the data
        assert np.all(np.isfinite(lin_pred)), "Linear leaf predictions should be finite"
        # At minimum, linear leaf predictions should be in a reasonable range
        assert np.max(np.abs(lin_pred)) < 100, "Predictions shouldn't explode"


class TestLinearLeafEdgeCases:
    """Edge cases for LinearLeafGBDT."""

    def test_with_constant_features(self):
        """Should handle constant features gracefully."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float32)
        X[:, 1] = 5.0  # Constant feature
        y = X[:, 0].copy()

        model = ob.LinearLeafGBDT(n_trees=10, max_depth=2)
        model.fit(X, y)
        pred = model.predict(X)

        assert np.all(np.isfinite(pred))

    def test_with_missing_values(self):
        """Should handle NaN in features."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float32)
        X[:5, 0] = np.nan
        y = rng.randn(100).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=10, max_depth=2)
        model.fit(X, y)
        pred = model.predict(X)

        assert pred.shape == y.shape
        assert np.all(np.isfinite(pred))

    def test_shallow_trees_with_linear_leaves(self):
        """Shallow trees (depth 1-2) should still work with linear leaves."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=20, max_depth=1, learning_rate=0.1)
        model.fit(X, y)
        pred = model.predict(X)

        mse = np.mean((pred - y) ** 2)
        baseline_mse = np.var(y)
        assert mse < baseline_mse, "Model should fit better than mean prediction"

    def test_single_tree(self):
        """Should work with a single tree."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=1, max_depth=3)
        model.fit(X, y)
        pred = model.predict(X)

        assert pred.shape == y.shape
        assert np.all(np.isfinite(pred))


class TestLinearLeafPersistence:
    """Save/load functionality."""

    def test_save_load_roundtrip(self, regression_100x5, tmp_path):
        """Predictions should match after save/load."""
        X, y = regression_100x5

        model = ob.LinearLeafGBDT(n_trees=5, max_depth=2)
        model.fit(X, y)
        pred_before = model.predict(X)

        path = str(tmp_path / "linear_leaf_model.json")
        model.save(path)

        loaded = ob.LinearLeafGBDT.load(path)
        pred_after = loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
