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


# =============================================================================
# Growth strategy tests (leafwise vs XGBoost lossguide, symmetric invariants)
# =============================================================================

def _growth_regression_data(n=6000, n_features=10, seed=7):
    """Fixed-seed nonlinear regression data with an 80/20 train/test split."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    y = (
        1.5 * X[:, 0]
        + np.sin(2.0 * X[:, 1])
        + 0.5 * X[:, 2] * X[:, 3]
        - 0.3 * X[:, 4]
        + 0.1 * rng.randn(n)
    ).astype(np.float32)
    n_train = int(0.8 * n)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _growth_binary_data(n=6000, n_features=10, seed=11):
    """Fixed-seed nonlinear binary data with an 80/20 train/test split."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features).astype(np.float32)
    logits = (
        1.2 * X[:, 0]
        + np.sin(2.0 * X[:, 1])
        + 0.5 * X[:, 2] * X[:, 3]
        - 0.4 * X[:, 4]
    )
    y = (logits + 0.3 * rng.randn(n) > 0).astype(np.float32)
    n_train = int(0.8 * n)
    return X[:n_train], X[n_train:], y[:n_train], y[n_train:]


def _matched_leafwise_params(n_trees=50, max_leaves=16, max_depth=12):
    """Hyperparameters aligning OpenBoost leafwise with XGBoost lossguide.

    Same conventions as _matched_params (matched eta/reg_lambda/
    min_child_weight/max_bin), plus best-first growth with a shared leaf
    budget. max_depth is set high on both sides so max_leaves is the
    binding constraint.
    """
    return dict(
        ob_params=dict(
            n_trees=n_trees,
            max_depth=max_depth,
            learning_rate=0.1,
            reg_lambda=1.0,
            min_child_weight=1.0,
            subsample=1.0,
            colsample_bytree=1.0,
            growth='leafwise',
            max_leaves=max_leaves,
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
            grow_policy='lossguide',
            max_leaves=max_leaves,
            max_bin=255,
            random_state=42,
        ),
    )


@pytest.fixture(scope="module")
def leafwise_regression_preds():
    """Test-set predictions from matched leafwise OB / lossguide XGB regressors."""
    X_train, X_test, y_train, y_test = _growth_regression_data()
    params = _matched_leafwise_params()

    ob_model = ob.GradientBoosting(**params['ob_params'])
    ob_model.fit(X_train, y_train)
    ob_pred = ob_model.predict(X_test)

    xgb_model = xgb.XGBRegressor(**params['xgb_params'])
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)

    return dict(ob_pred=ob_pred, xgb_pred=xgb_pred,
                y_train=y_train, y_test=y_test)


@pytest.fixture(scope="module")
def leafwise_binary_preds():
    """Test-set probabilities from matched leafwise OB / lossguide XGB classifiers."""
    X_train, X_test, y_train, y_test = _growth_binary_data()
    params = _matched_leafwise_params()

    ob_model = ob.GradientBoosting(loss='logloss', **params['ob_params'])
    ob_model.fit(X_train, y_train)
    ob_prob = 1.0 / (1.0 + np.exp(-ob_model.predict(X_test)))

    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        **params['xgb_params'],
    )
    xgb_model.fit(X_train, y_train)
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

    return dict(ob_prob=ob_prob, xgb_prob=xgb_prob, y_test=y_test)


@pytest.mark.xgboost
class TestLeafwiseLossguideAgreement:
    """OpenBoost growth='leafwise' vs XGBoost grow_policy='lossguide'."""

    def test_regression_predictions_close(self, leafwise_regression_preds):
        """Test-set predictions should be within 10% relative RMSE."""
        p = leafwise_regression_preds
        rmse_diff = np.sqrt(np.mean((p['ob_pred'] - p['xgb_pred']) ** 2))
        rmse_target = np.sqrt(np.mean((p['y_test'] - np.mean(p['y_train'])) ** 2))

        assert rmse_diff / rmse_target < 0.10, (
            f"Prediction RMSE diff {rmse_diff:.4f} is >10% of target RMSE {rmse_target:.4f}"
        )

    def test_regression_same_direction(self, leafwise_regression_preds):
        """Predictions should agree on relative ordering (correlation > 0.95)."""
        p = leafwise_regression_preds
        correlation = np.corrcoef(p['ob_pred'], p['xgb_pred'])[0, 1]
        assert correlation > 0.95, (
            f"Prediction correlation should be > 0.95, got {correlation:.4f}"
        )

    def test_regression_rmse_parity(self, leafwise_regression_preds):
        """Held-out RMSE should be within 15% of XGBoost's."""
        p = leafwise_regression_preds
        ob_rmse = np.sqrt(np.mean((p['ob_pred'] - p['y_test']) ** 2))
        xgb_rmse = np.sqrt(np.mean((p['xgb_pred'] - p['y_test']) ** 2))

        assert ob_rmse < xgb_rmse * 1.15, (
            f"OB RMSE ({ob_rmse:.4f}) > 1.15x XGB RMSE ({xgb_rmse:.4f})"
        )

    def test_binary_probabilities_close(self, leafwise_binary_preds):
        """Test-set probabilities should be within 0.10 of each other on average."""
        p = leafwise_binary_preds
        mean_diff = np.mean(np.abs(p['ob_prob'] - p['xgb_prob']))
        assert mean_diff < 0.10, (
            f"Mean probability difference {mean_diff:.4f} > 0.10"
        )

    def test_binary_accuracy_comparable(self, leafwise_binary_preds):
        """Held-out accuracies should be within 5 percentage points."""
        p = leafwise_binary_preds
        ob_acc = np.mean((p['ob_prob'] > 0.5).astype(float) == p['y_test'])
        xgb_acc = np.mean((p['xgb_prob'] > 0.5).astype(float) == p['y_test'])

        assert abs(ob_acc - xgb_acc) < 0.05, (
            f"Accuracy gap too large: OB={ob_acc:.3f}, XGB={xgb_acc:.3f}"
        )


# =============================================================================
# Symmetric growth invariants (no reference library exists; property tests)
# =============================================================================

@pytest.fixture(scope="module")
def symmetric_regression_model():
    """Symmetric-growth model plus its train/test data."""
    X_train, X_test, y_train, y_test = _growth_regression_data()
    model = ob.GradientBoosting(
        n_trees=30, max_depth=5, learning_rate=0.1,
        reg_lambda=1.0, min_child_weight=1.0, growth='symmetric',
    )
    model.fit(X_train, y_train)
    return model, X_test, y_test


def _tree_leaf_values(tree):
    """Raw per-node value array of a TreeStructure (plain ndarray leaves)."""
    return np.asarray(tree.values, dtype=np.float32)


class TestSymmetricGrowthInvariants:
    """Structural and numerical properties of growth='symmetric' trees."""

    def test_one_split_per_level(self, symmetric_regression_model):
        """Every depth level of every tree uses exactly one (feature, threshold).

        Checks both representations of the oblivious structure:
        the level arrays (level_features/level_thresholds) and the standard
        node arrays (complete binary tree, children at 2i+1 / 2i+2).
        """
        model, _, _ = symmetric_regression_model
        assert len(model.trees_) == 30

        for t_idx, tree in enumerate(model.trees_):
            assert tree.is_symmetric, f"tree {t_idx} lost is_symmetric flag"
            assert tree.level_features is not None
            assert tree.level_thresholds is not None
            assert len(tree.level_features) == tree.depth
            assert len(tree.level_thresholds) == tree.depth
            # Node arrays cover the complete binary tree of this depth
            assert tree.n_nodes == 2 ** (tree.depth + 1) - 1

            for d in range(tree.depth):
                start, end = 2 ** d - 1, 2 ** (d + 1) - 1
                level_feats = tree.features[start:end]
                level_thrs = tree.thresholds[start:end]

                # Exactly one (feature, threshold) pair used across the level
                pairs = set(zip(level_feats.tolist(), level_thrs.tolist(), strict=True))
                assert len(pairs) == 1, (
                    f"tree {t_idx} depth {d} uses {len(pairs)} distinct splits: {pairs}"
                )
                # ... and it is the pair recorded in the level arrays
                assert pairs == {(int(tree.level_features[d]),
                                  int(tree.level_thresholds[d]))}
                assert int(tree.level_features[d]) >= 0

                # Children follow the complete-binary-tree layout
                nodes = np.arange(start, end)
                np.testing.assert_array_equal(tree.left_children[start:end], 2 * nodes + 1)
                np.testing.assert_array_equal(tree.right_children[start:end], 2 * nodes + 2)

            # The last level is all leaves
            leaf_start = 2 ** tree.depth - 1
            assert np.all(tree.features[leaf_start:tree.n_nodes] == -1)
            assert np.all(tree.left_children[leaf_start:tree.n_nodes] == -1)
            assert np.all(tree.right_children[leaf_start:tree.n_nodes] == -1)

        # Learnable data: the ensemble must actually split
        assert any(tree.depth >= 1 for tree in model.trees_)

    def test_hand_derived_predictions_match(self, symmetric_regression_model):
        """Routing 100 rows by hand through each tree reproduces model.predict.

        Derives predictions two independent ways from the inspected structure:
        (a) oblivious routing via level_features/level_thresholds bit tricks,
        (b) standard node-array traversal via features/left/right children.
        Both must match model.predict().
        """
        model, X_test, _ = symmetric_regression_model
        X_sub = X_test[:100]
        binned = np.asarray(model.X_binned_.transform(X_sub).data)
        n = X_sub.shape[0]

        lr = np.float32(model.learning_rate)
        pred_levels = np.full(n, model.base_score_, dtype=np.float32)
        pred_nodes = np.full(n, model.base_score_, dtype=np.float32)

        for tree in model.trees_:
            values = _tree_leaf_values(tree)

            # (a) Oblivious routing: leaf_id built one bit per level
            leaf_ids = np.zeros(n, dtype=np.int64)
            for d in range(tree.depth):
                feat = int(tree.level_features[d])
                thr = int(tree.level_thresholds[d])
                leaf_ids = 2 * leaf_ids + (binned[feat, :] > thr).astype(np.int64)
            leaf_start = 2 ** tree.depth - 1
            pred_levels += lr * values[leaf_start + leaf_ids]

            # (b) Standard traversal over the node arrays
            tree_vals = np.empty(n, dtype=np.float32)
            for i in range(n):
                node = 0
                while tree.left_children[node] != -1:
                    feat = int(tree.features[node])
                    thr = int(tree.thresholds[node])
                    if binned[feat, i] <= thr:
                        node = int(tree.left_children[node])
                    else:
                        node = int(tree.right_children[node])
                tree_vals[i] = values[node]
            pred_nodes += lr * tree_vals

        model_pred = model.predict(X_sub)
        np.testing.assert_allclose(pred_levels, model_pred, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(pred_nodes, model_pred, rtol=1e-4, atol=1e-4)

    def test_symmetric_r2_sane(self, symmetric_regression_model):
        """Symmetric growth must actually learn (held-out R2 > 0.5)."""
        model, X_test, y_test = symmetric_regression_model
        pred = model.predict(X_test)
        ss_res = np.sum((y_test - pred) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2 = 1 - ss_res / ss_tot
        assert r2 > 0.5, f"Symmetric growth R2 {r2:.4f} <= 0.5 on learnable data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
