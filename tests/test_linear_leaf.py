"""Tests for LinearLeafGBDT model.

Verifies that gradient boosting with linear models in leaves works correctly
on CPU. These are the first CPU tests for this model variant.
"""

import numpy as np
import pytest

import openboost as ob


def _reference_predict(model, X):
    """Golden reference: per-sample Python loop predict.

    This mirrors, operation for operation (including float32 accumulation
    order), the original O(n_samples) implementation of
    ``LinearLeafTree.predict`` / ``LinearLeafGBDT.predict`` that predated
    vectorization. The vectorized implementation must match it exactly.
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples = X.shape[0]

    base = getattr(model, "base_score_", np.float32(0.0))
    pred = np.full(n_samples, base, dtype=np.float32)

    for tree in model.trees_:
        if tree.training_binned is not None:
            binned = tree.training_binned.transform(X).data
        else:
            binned = ob.array(X).data
        binned = np.asarray(binned)

        ts = tree.tree_structure
        tree_pred = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_samples):
            node = 0
            while ts.left_children[node] != -1:
                if binned[ts.features[node], i] <= ts.thresholds[node]:
                    node = ts.left_children[node]
                else:
                    node = ts.right_children[node]

            leaf_idx = tree.leaf_ids.get(int(node), 0)
            weights = tree.leaf_weights[leaf_idx]
            feat_indices = tree.leaf_features[leaf_idx]

            p = weights[0]
            for j, feat_idx in enumerate(feat_indices):
                if j + 1 < len(weights):
                    p = p + weights[j + 1] * X[i, feat_idx]
            tree_pred[i] = p

        pred = pred + model.learning_rate * tree_pred

    return pred


class TestLinearLeafVectorizedPredict:
    """Vectorized predict must match the original per-sample loop exactly."""

    @pytest.mark.parametrize("max_features_linear", ["sqrt", None, 2])
    def test_predict_matches_reference_loop(self, max_features_linear):
        """Vectorized predictions equal the per-sample reference within 1e-10."""
        rng = np.random.RandomState(7)
        X = rng.randn(300, 8).astype(np.float32)
        y = (2 * X[:, 0] - X[:, 3] + 0.5 * rng.randn(300)).astype(np.float32)

        model = ob.LinearLeafGBDT(
            n_trees=12, max_depth=3, max_features_linear=max_features_linear
        )
        model.fit(X, y)

        # Includes values outside the training range (extrapolation bins)
        X_new = (rng.randn(500, 8) * 2.0).astype(np.float32)
        np.testing.assert_allclose(
            model.predict(X_new), _reference_predict(model, X_new),
            rtol=0, atol=1e-10,
        )

    def test_predict_matches_reference_with_missing(self):
        """Equivalence must hold when prediction data contains NaN."""
        rng = np.random.RandomState(11)
        X = rng.randn(200, 5).astype(np.float32)
        y = (X[:, 0] + 0.5 * X[:, 1]).astype(np.float32)

        model = ob.LinearLeafGBDT(n_trees=8, max_depth=3)
        model.fit(X, y)

        X_new = rng.randn(300, 5).astype(np.float32)
        X_new[::7, 2] = np.nan  # NaN routes through MISSING_BIN
        np.testing.assert_allclose(
            model.predict(X_new), _reference_predict(model, X_new),
            rtol=0, atol=1e-10,
        )


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
        """LinearLeaf must beat standard GBDT decisively out of the training range.

        Standard GBDT predicts piecewise constants, so beyond the training
        range it saturates at the boundary value. Linear leaves extrapolate
        the local trend, so on y = 3x their out-of-range RMSE should be a
        small fraction of standard GBDT's.
        """
        rng = np.random.RandomState(42)
        # Training: x in [-2, 2], y = 3x + small noise
        X_train = rng.uniform(-2, 2, (400, 1)).astype(np.float32)
        y_train = (3 * X_train[:, 0] + 0.1 * rng.randn(400)).astype(np.float32)

        # Test: x in [3, 5], well outside the training range
        X_test = rng.uniform(3, 5, (100, 1)).astype(np.float32)
        y_test = (3 * X_test[:, 0]).astype(np.float32)

        standard = ob.GradientBoosting(n_trees=50, max_depth=4, learning_rate=0.1)
        standard.fit(X_train, y_train)
        std_rmse = float(np.sqrt(np.mean((standard.predict(X_test) - y_test) ** 2)))

        linear = ob.LinearLeafGBDT(n_trees=50, max_depth=3, learning_rate=0.1)
        linear.fit(X_train, y_train)
        lin_pred = linear.predict(X_test)
        lin_rmse = float(np.sqrt(np.mean((lin_pred - y_test) ** 2)))

        assert np.all(np.isfinite(lin_pred)), "Linear leaf predictions should be finite"
        # Standard GBDT saturates near y = 6 (boundary), giving RMSE ~6 here;
        # linear leaves should track y = 3x closely. Factor 2 leaves margin.
        assert lin_rmse <= 0.5 * std_rmse, (
            f"LinearLeafGBDT should extrapolate at least 2x better: "
            f"linear RMSE {lin_rmse:.4f} vs standard RMSE {std_rmse:.4f}"
        )


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
