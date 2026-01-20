"""Tests for input validation (Phase 20.3).

Tests that helpful error messages are raised for common user mistakes.
"""

import numpy as np
import pytest
import warnings


class TestValidateX:
    """Test X array validation."""

    def test_wrong_type_raises(self):
        """Test that wrong type raises with helpful error."""
        import openboost as ob

        model = ob.GradientBoosting(n_trees=5)

        # String gets converted to 0D array, which fails dimension check
        with pytest.raises(ValueError, match="2D|shape"):
            model.fit("not an array", [1, 2, 3])

    def test_1d_array_raises(self):
        """Test that 1D array raises with helpful message."""
        import openboost as ob

        X = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        y = np.array([1, 2, 3, 4, 5], dtype=np.float32)

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="reshape"):
            model.fit(X, y)

    def test_empty_array_raises(self):
        """Test that empty array raises with helpful message."""
        import openboost as ob

        X = np.empty((0, 5), dtype=np.float32)
        y = np.empty(0, dtype=np.float32)

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="empty"):
            model.fit(X, y)

    def test_nan_handled_by_binning(self):
        """Test that NaN values are handled by OpenBoost's binning.
        
        OpenBoost's array() function handles NaN values natively,
        so fit() accepts NaN in X (they get binned to a special missing bin).
        """
        import openboost as ob

        X = np.array([[1, 2], [np.nan, 4], [5, 6]], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.float32)

        # Increase depth to avoid hyperparameter warning
        model = ob.GradientBoosting(n_trees=5, max_depth=2, min_child_weight=0.5)

        # Should NOT raise - OpenBoost handles NaN natively
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)

        # Predict should also work
        pred = model.predict(X)
        assert len(pred) == 3

    def test_inf_raises(self):
        """Test that infinite values raise."""
        import openboost as ob

        X = np.array([[1, 2], [np.inf, 4], [5, 6]], dtype=np.float32)
        y = np.array([1, 2, 3], dtype=np.float32)

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="infinite"):
            model.fit(X, y)

    def test_dtype_conversion_warns(self):
        """Test that non-float dtype triggers warning."""
        import openboost as ob

        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int32)
        y = np.array([1, 2, 3], dtype=np.float32)

        model = ob.GradientBoosting(n_trees=5, max_depth=2)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)
            # Should have conversion warning
            dtype_warnings = [x for x in w if "dtype" in str(x.message).lower()]
            assert len(dtype_warnings) > 0


class TestValidateY:
    """Test y array validation."""

    def test_shape_mismatch_raises(self):
        """Test that X/y shape mismatch raises clear error."""
        import openboost as ob

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(50).astype(np.float32)  # Wrong size

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="inconsistent"):
            model.fit(X, y)

    def test_nan_in_y_raises(self):
        """Test that NaN in y raises."""
        import openboost as ob

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        y[10] = np.nan

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="NaN"):
            model.fit(X, y)

    def test_inf_in_y_raises(self):
        """Test that infinity in y raises."""
        import openboost as ob

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        y[10] = np.inf

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="infinite"):
            model.fit(X, y)

    def test_2d_y_with_single_column_works(self):
        """Test that 2D y with shape (n, 1) is flattened."""
        import openboost as ob

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100, 1).astype(np.float32)  # 2D but single column

        model = ob.GradientBoosting(n_trees=5, max_depth=2)
        model.fit(X, y)  # Should work

        assert model.predict(X).shape == (100,)


class TestValidateHyperparameters:
    """Test hyperparameter validation."""

    def test_negative_n_trees_raises(self):
        """Test that negative n_trees raises."""
        import openboost as ob

        with pytest.raises(ValueError, match="n_trees"):
            model = ob.GradientBoosting(n_trees=-1)
            X = np.random.randn(100, 5).astype(np.float32)
            y = np.random.randn(100).astype(np.float32)
            model.fit(X, y)

    def test_negative_learning_rate_raises(self):
        """Test that negative learning_rate raises."""
        import openboost as ob

        with pytest.raises(ValueError, match="learning_rate"):
            model = ob.GradientBoosting(n_trees=5, learning_rate=-0.1)
            X = np.random.randn(100, 5).astype(np.float32)
            y = np.random.randn(100).astype(np.float32)
            model.fit(X, y)

    def test_invalid_subsample_raises(self):
        """Test that subsample outside (0, 1] raises."""
        import openboost as ob

        with pytest.raises(ValueError, match="subsample"):
            model = ob.GradientBoosting(n_trees=5, subsample=0.0)
            X = np.random.randn(100, 5).astype(np.float32)
            y = np.random.randn(100).astype(np.float32)
            model.fit(X, y)

        with pytest.raises(ValueError, match="subsample"):
            model = ob.GradientBoosting(n_trees=5, subsample=1.5)
            model.fit(X, y)

    def test_high_learning_rate_warns(self):
        """Test that very high learning rate warns."""
        import openboost as ob

        model = ob.GradientBoosting(n_trees=5, max_depth=2, learning_rate=2.0)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)
            lr_warnings = [x for x in w if "learning_rate" in str(x.message)]
            assert len(lr_warnings) > 0

    def test_deep_trees_warns(self):
        """Test that very deep trees warn."""
        import openboost as ob

        model = ob.GradientBoosting(n_trees=5, max_depth=20)
        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.fit(X, y)
            depth_warnings = [x for x in w if "max_depth" in str(x.message)]
            assert len(depth_warnings) > 0


class TestValidatePredict:
    """Test predict validation."""

    def test_predict_before_fit_raises(self):
        """Test that predict before fit raises clear error."""
        import openboost as ob

        model = ob.GradientBoosting(n_trees=5)
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(ValueError, match="not fitted"):
            model.predict(X)

    def test_wrong_n_features_raises(self):
        """Test that wrong number of features raises clear error."""
        import openboost as ob

        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)

        model = ob.GradientBoosting(n_trees=5, max_depth=2)
        model.fit(X_train, y_train)

        X_test = np.random.randn(10, 3).astype(np.float32)  # Wrong features

        with pytest.raises(ValueError, match="features"):
            model.predict(X_test)


class TestValidateSampleWeight:
    """Test sample weight validation."""

    def test_wrong_length_raises(self):
        """Test that wrong length sample_weight raises."""
        import openboost as ob

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        weights = np.ones(50).astype(np.float32)  # Wrong length

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="elements"):
            model.fit(X, y, sample_weight=weights)

    def test_negative_weights_raise(self):
        """Test that negative weights raise."""
        import openboost as ob

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)
        weights = np.ones(100).astype(np.float32)
        weights[10] = -1

        model = ob.GradientBoosting(n_trees=5)

        with pytest.raises(ValueError, match="negative"):
            model.fit(X, y, sample_weight=weights)


class TestValidateEvalSet:
    """Test eval_set validation."""

    def test_wrong_format_raises(self):
        """Test that wrong eval_set format raises."""
        import openboost as ob
        from openboost import EarlyStopping

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = ob.GradientBoosting(n_trees=100)

        # eval_set should be list of tuples
        with pytest.raises(TypeError, match="list"):
            model.fit(X, y, callbacks=[EarlyStopping()], eval_set=(X, y))

    def test_wrong_n_features_raises(self):
        """Test that eval_set with wrong features raises."""
        import openboost as ob
        from openboost import EarlyStopping

        X_train = np.random.randn(100, 5).astype(np.float32)
        y_train = np.random.randn(100).astype(np.float32)

        X_val = np.random.randn(20, 3).astype(np.float32)  # Wrong features
        y_val = np.random.randn(20).astype(np.float32)

        model = ob.GradientBoosting(n_trees=100)

        with pytest.raises(ValueError, match="features"):
            model.fit(
                X_train, y_train,
                callbacks=[EarlyStopping()],
                eval_set=[(X_val, y_val)]
            )


class TestValidationIntegration:
    """Integration tests for validation."""

    def test_valid_input_works(self):
        """Test that valid inputs work correctly."""
        import openboost as ob

        X = np.random.randn(100, 5).astype(np.float32)
        y = np.random.randn(100).astype(np.float32)

        model = ob.GradientBoosting(n_trees=10, max_depth=3)
        model.fit(X, y)

        pred = model.predict(X)
        assert pred.shape == (100,)
        assert not np.any(np.isnan(pred))

    def test_list_input_converted(self):
        """Test that list inputs are converted to arrays."""
        import openboost as ob

        X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
        y = [1, 2, 3, 4, 5]

        model = ob.GradientBoosting(n_trees=5, max_depth=2)

        # Should convert lists to arrays with warning
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            model.fit(X, y)

        pred = model.predict(X)
        assert len(pred) == 5
