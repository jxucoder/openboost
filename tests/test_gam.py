"""Tests for OpenBoostGAM model.

Verifies that the GPU-accelerated Generalized Additive Model works
correctly on CPU. These are the first CPU tests for this model variant.
"""

import numpy as np
import pytest

import openboost as ob


class TestGAMBasic:
    """Basic functionality tests."""

    def test_basic_fit_predict(self, regression_200x10):
        """Fit and predict should produce correct shapes."""
        X, y = regression_200x10

        gam = ob.OpenBoostGAM(n_rounds=50, learning_rate=0.05, reg_lambda=1.0)
        gam.fit(X, y)
        pred = gam.predict(X)

        assert pred.shape == y.shape
        assert pred.dtype == np.float32
        assert np.all(np.isfinite(pred))

    def test_shape_values_shape(self, regression_200x10):
        """shape_values_ should be (n_features, 256)."""
        X, y = regression_200x10

        gam = ob.OpenBoostGAM(n_rounds=20, learning_rate=0.05)
        gam.fit(X, y)

        assert gam.shape_values_ is not None
        assert gam.shape_values_.shape == (10, 256), (
            f"Expected shape (10, 256), got {gam.shape_values_.shape}"
        )

    def test_training_reduces_loss(self, regression_200x10):
        """Training should reduce loss compared to baseline."""
        X, y = regression_200x10

        gam = ob.OpenBoostGAM(n_rounds=100, learning_rate=0.05)
        gam.fit(X, y)
        pred = gam.predict(X)

        mse = np.mean((pred - y) ** 2)
        baseline_mse = np.var(y)

        assert mse < baseline_mse * 0.5, (
            f"GAM MSE ({mse:.4f}) should be well below baseline ({baseline_mse:.4f})"
        )

    def test_deterministic(self, regression_100x5):
        """Same input should produce identical output."""
        X, y = regression_100x5

        gam1 = ob.OpenBoostGAM(n_rounds=20, learning_rate=0.05)
        gam1.fit(X, y)
        pred1 = gam1.predict(X)

        gam2 = ob.OpenBoostGAM(n_rounds=20, learning_rate=0.05)
        gam2.fit(X, y)
        pred2 = gam2.predict(X)

        np.testing.assert_array_equal(pred1, pred2)


class TestGAMInterpretability:
    """Verify GAM interpretability properties."""

    def test_shape_functions_capture_correct_features(self):
        """When y = f(X[:,0]), feature 0's shape function should be most active."""
        rng = np.random.RandomState(42)
        X = rng.randn(300, 5).astype(np.float32)
        y = np.sin(X[:, 0]).astype(np.float32)

        gam = ob.OpenBoostGAM(n_rounds=500, learning_rate=0.03)
        gam.fit(X, y)

        # Feature 0 should have the largest shape function range
        ranges = [np.ptp(gam.shape_values_[f]) for f in range(5)]
        assert np.argmax(ranges) == 0, (
            f"Feature 0 should have largest range but ranges are: {ranges}"
        )

    def test_additive_prediction_structure(self, regression_100x5):
        """Predictions should be sum of shape functions + base score."""
        X, y = regression_100x5

        gam = ob.OpenBoostGAM(n_rounds=30, learning_rate=0.05)
        gam.fit(X, y)

        # Get predictions the normal way
        pred_normal = gam.predict(X)

        # Manually compute from shape functions
        binned = gam.X_binned_
        binned_data = binned.data
        if hasattr(binned_data, 'copy_to_host'):
            binned_data = binned_data.copy_to_host()
        binned_data = np.asarray(binned_data)

        base = getattr(gam, 'base_score_', np.float32(0.0))
        pred_manual = np.full(len(y), base, dtype=np.float32)
        for f in range(X.shape[1]):
            pred_manual += gam.shape_values_[f, binned_data[f, :]]

        np.testing.assert_allclose(pred_normal, pred_manual, atol=1e-5)


class TestGAMClassification:
    """GAM with classification loss."""

    def test_logloss(self, binary_500x10):
        """GAM should work with logloss for binary classification."""
        X, y = binary_500x10

        gam = ob.OpenBoostGAM(n_rounds=100, learning_rate=0.05, loss='logloss')
        gam.fit(X, y)
        pred_raw = gam.predict(X)

        # Convert to probabilities
        prob = 1.0 / (1.0 + np.exp(-pred_raw))
        labels = (prob > 0.5).astype(float)
        accuracy = np.mean(labels == y)

        assert accuracy > 0.70, f"GAM classification accuracy {accuracy:.3f} < 0.70"


class TestGAMEdgeCases:
    """Edge cases for OpenBoostGAM."""

    def test_predict_before_fit_raises(self):
        """Predict on unfitted model should raise."""
        gam = ob.OpenBoostGAM(n_rounds=10)
        rng = np.random.RandomState(42)
        X = rng.randn(10, 3).astype(np.float32)

        with pytest.raises(RuntimeError, match="not fitted"):
            gam.predict(X)

    def test_single_round(self):
        """Should work with a single boosting round."""
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3).astype(np.float32)
        y = rng.randn(50).astype(np.float32)

        gam = ob.OpenBoostGAM(n_rounds=1, learning_rate=0.1)
        gam.fit(X, y)
        pred = gam.predict(X)

        assert pred.shape == y.shape
        assert np.all(np.isfinite(pred))

    def test_constant_target(self):
        """GAM with constant target should predict that constant."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float32)
        y = np.full(100, 2.5, dtype=np.float32)

        gam = ob.OpenBoostGAM(n_rounds=50, learning_rate=0.1)
        gam.fit(X, y)
        pred = gam.predict(X)

        np.testing.assert_allclose(pred, 2.5, atol=0.2,
                                   err_msg="Should converge to constant target")


class TestGAMPersistence:
    """Save/load functionality."""

    def test_save_load_roundtrip(self, regression_100x5, tmp_path):
        """Predictions should match after save/load."""
        X, y = regression_100x5

        gam = ob.OpenBoostGAM(n_rounds=10, learning_rate=0.05)
        gam.fit(X, y)
        pred_before = gam.predict(X)

        path = str(tmp_path / "gam_model.json")
        gam.save(path)

        loaded = ob.OpenBoostGAM.load(path)
        pred_after = loaded.predict(X)

        np.testing.assert_array_equal(pred_before, pred_after)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
