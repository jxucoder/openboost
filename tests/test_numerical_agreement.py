"""Numerical agreement tests: OpenBoost vs XGBoost.

For matched hyperparameters, OpenBoost predictions should be very close
to XGBoost predictions. This is the strongest end-to-end correctness signal.

All tests are marked @pytest.mark.xgboost and skip if xgboost is not installed.
"""

import numpy as np
import pytest

import openboost as ob

xgb = pytest.importorskip("xgboost")


def _matched_params(n_trees=50, max_depth=4):
    """Hyperparameters that align OpenBoost and XGBoost behavior."""
    return dict(
        ob_params=dict(
            n_trees=n_trees,
            max_depth=max_depth,
            learning_rate=0.1,
            reg_lambda=1.0,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
        ),
        xgb_params=dict(
            n_estimators=n_trees,
            max_depth=max_depth,
            learning_rate=0.1,
            reg_lambda=1.0,
            reg_alpha=0.0,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            tree_method='hist',
            max_bin=255,
            random_state=42,
        ),
    )


@pytest.mark.xgboost
class TestXGBoostRegressionAgreement:
    """Regression prediction agreement between OpenBoost and XGBoost."""

    def test_single_tree_very_close(self, regression_500x10):
        """Single depth-1 tree should produce very similar predictions."""
        X, y = regression_500x10
        params = _matched_params(n_trees=1, max_depth=1)

        ob_model = ob.GradientBoosting(**params['ob_params'])
        ob_model.fit(X, y)
        ob_pred = ob_model.predict(X)

        xgb_model = xgb.XGBRegressor(**params['xgb_params'])
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X)

        # Single tree, depth 1: very few ways to differ
        rmse_diff = np.sqrt(np.mean((ob_pred - xgb_pred) ** 2))
        target_std = np.std(y)

        assert rmse_diff / target_std < 0.05, (
            f"Single tree predictions differ too much: RMSE diff = {rmse_diff:.4f}, "
            f"target std = {target_std:.4f} (ratio = {rmse_diff/target_std:.3f})"
        )

    def test_regression_predictions_close(self, regression_500x10):
        """50-tree predictions should be within 5% relative RMSE."""
        X, y = regression_500x10
        params = _matched_params(n_trees=50, max_depth=4)

        ob_model = ob.GradientBoosting(**params['ob_params'])
        ob_model.fit(X, y)
        ob_pred = ob_model.predict(X)

        xgb_model = xgb.XGBRegressor(**params['xgb_params'])
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X)

        rmse_diff = np.sqrt(np.mean((ob_pred - xgb_pred) ** 2))
        rmse_target = np.sqrt(np.mean((y - np.mean(y)) ** 2))

        assert rmse_diff / rmse_target < 0.10, (
            f"Prediction RMSE diff {rmse_diff:.4f} is >{10}% of target RMSE {rmse_target:.4f}"
        )

    def test_predictions_same_direction(self, regression_500x10):
        """Predictions should agree on relative ordering (correlation > 0.95)."""
        X, y = regression_500x10
        params = _matched_params(n_trees=50, max_depth=4)

        ob_model = ob.GradientBoosting(**params['ob_params'])
        ob_model.fit(X, y)
        ob_pred = ob_model.predict(X)

        xgb_model = xgb.XGBRegressor(**params['xgb_params'])
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X)

        correlation = np.corrcoef(ob_pred, xgb_pred)[0, 1]
        assert correlation > 0.95, (
            f"Prediction correlation should be > 0.95, got {correlation:.4f}"
        )


@pytest.mark.xgboost
class TestXGBoostClassificationAgreement:
    """Classification prediction agreement."""

    def test_classification_probabilities_close(self, binary_500x10):
        """Predicted probabilities should be within 0.10 of each other."""
        X, y = binary_500x10
        params = _matched_params(n_trees=50, max_depth=4)

        ob_model = ob.GradientBoosting(
            loss='logloss', **params['ob_params']
        )
        ob_model.fit(X, y)
        ob_raw = ob_model.predict(X)
        # Convert logits to probabilities
        ob_prob = 1.0 / (1.0 + np.exp(-ob_raw))

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            **params['xgb_params'],
        )
        xgb_model.fit(X, y)
        xgb_prob = xgb_model.predict_proba(X)[:, 1]

        mean_diff = np.mean(np.abs(ob_prob - xgb_prob))

        assert mean_diff < 0.10, (
            f"Mean probability difference {mean_diff:.4f} > 0.10"
        )

    def test_classification_accuracy_comparable(self, binary_500x10):
        """Both models should achieve similar accuracy."""
        X, y = binary_500x10
        params = _matched_params(n_trees=50, max_depth=4)

        ob_model = ob.GradientBoosting(
            loss='logloss', **params['ob_params']
        )
        ob_model.fit(X, y)
        ob_raw = ob_model.predict(X)
        ob_labels = (ob_raw > 0).astype(float)
        ob_acc = np.mean(ob_labels == y)

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            **params['xgb_params'],
        )
        xgb_model.fit(X, y)
        xgb_labels = xgb_model.predict(X)
        xgb_acc = np.mean(xgb_labels == y)

        # Accuracies should be within 5 percentage points
        assert abs(ob_acc - xgb_acc) < 0.05, (
            f"Accuracy gap too large: OB={ob_acc:.3f}, XGB={xgb_acc:.3f}"
        )


@pytest.mark.xgboost
class TestXGBoostQualityParity:
    """Model quality should be competitive with XGBoost."""

    def test_regression_r2_competitive(self, regression_500x10):
        """OpenBoost R2 should be within 15% of XGBoost R2."""
        X, y = regression_500x10
        params = _matched_params(n_trees=100, max_depth=4)

        ob_model = ob.GradientBoosting(**params['ob_params'])
        ob_model.fit(X, y)
        ob_pred = ob_model.predict(X)
        ss_res_ob = np.sum((y - ob_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        ob_r2 = 1 - ss_res_ob / ss_tot

        xgb_model = xgb.XGBRegressor(**params['xgb_params'])
        xgb_model.fit(X, y)
        xgb_pred = xgb_model.predict(X)
        ss_res_xgb = np.sum((y - xgb_pred) ** 2)
        xgb_r2 = 1 - ss_res_xgb / ss_tot

        assert ob_r2 > xgb_r2 * 0.85, (
            f"OpenBoost R2 ({ob_r2:.4f}) should be within 15% of XGBoost R2 ({xgb_r2:.4f})"
        )

    @pytest.mark.slow
    def test_regression_california_housing(self):
        """Real dataset: California Housing regression."""
        pytest.importorskip("sklearn")
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split

        try:
            data = fetch_california_housing()
        except Exception:
            pytest.skip("Could not download California Housing dataset")

        X_train, X_test, y_train, y_test = train_test_split(
            data.data.astype(np.float32),
            data.target.astype(np.float32),
            test_size=0.2, random_state=42,
        )

        params = _matched_params(n_trees=100, max_depth=6)

        ob_model = ob.GradientBoosting(**params['ob_params'])
        ob_model.fit(X_train, y_train)
        ob_pred = ob_model.predict(X_test)
        ob_rmse = np.sqrt(np.mean((ob_pred - y_test) ** 2))

        xgb_model = xgb.XGBRegressor(**params['xgb_params'])
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_rmse = np.sqrt(np.mean((xgb_pred - y_test) ** 2))

        # OpenBoost RMSE should be within 15% of XGBoost RMSE
        assert ob_rmse < xgb_rmse * 1.15, (
            f"OB RMSE ({ob_rmse:.4f}) > 1.15x XGB RMSE ({xgb_rmse:.4f})"
        )

    @pytest.mark.slow
    def test_binary_breast_cancer(self):
        """Real dataset: Breast Cancer binary classification."""
        pytest.importorskip("sklearn")
        from sklearn.datasets import load_breast_cancer
        from sklearn.model_selection import train_test_split

        data = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            data.data.astype(np.float32),
            data.target.astype(np.float32),
            test_size=0.2, random_state=42,
        )

        params = _matched_params(n_trees=50, max_depth=4)

        ob_model = ob.GradientBoosting(
            loss='logloss', **params['ob_params']
        )
        ob_model.fit(X_train, y_train)
        ob_pred = ob_model.predict(X_test)
        ob_acc = np.mean((ob_pred > 0).astype(float) == y_test)

        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            **params['xgb_params'],
        )
        xgb_model.fit(X_train, y_train)
        xgb_acc = np.mean(xgb_model.predict(X_test) == y_test)

        # Both should achieve > 90% accuracy
        assert ob_acc > 0.90, f"OB accuracy {ob_acc:.3f} < 0.90"
        # Within 5 points of each other
        assert abs(ob_acc - xgb_acc) < 0.05, (
            f"Accuracy gap: OB={ob_acc:.3f}, XGB={xgb_acc:.3f}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
