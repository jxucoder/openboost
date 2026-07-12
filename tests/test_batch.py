"""Correctness tests for train-many batch fitting."""

import numpy as np
import pytest

import openboost as ob


def _regression_data():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(64, 3)).astype(np.float32)
    y = (1.5 * X[:, 0] - 0.7 * X[:, 1] + 0.2 * X[:, 2]).astype(np.float32)
    return X, y


def _predict_ensemble(trees, X_binned, learning_rate):
    prediction = np.zeros(X_binned.n_samples, dtype=np.float32)
    for tree in trees:
        tree_prediction = tree(X_binned)
        if hasattr(tree_prediction, "copy_to_host"):
            tree_prediction = tree_prediction.copy_to_host()
        prediction += learning_rate * np.asarray(tree_prediction)
    return prediction


def test_batch_matches_independent_training():
    X, y = _regression_data()
    X_binned = ob.array(X, n_bins=254)
    configs = ob.ConfigBatch.from_lists(
        max_depths=[2, 3],
        reg_lambdas=[0.5, 1.0],
        min_child_weights=[1.0, 1.0],
        learning_rates=[0.1, 0.2],
        n_rounds=4,
    )

    batch_trees = ob.fit_trees_batch(X_binned, configs=configs, y=y)

    for config_index, config in enumerate(configs):
        expected_prediction = np.zeros_like(y)
        for _ in range(configs.n_rounds):
            grad, hess = ob.mse_gradient(expected_prediction, y)
            tree = ob.fit_tree(
                X_binned,
                grad,
                hess,
                max_depth=config["max_depth"],
                min_child_weight=config["min_child_weight"],
                reg_lambda=config["reg_lambda"],
            )
            expected_prediction += config["learning_rate"] * tree(X_binned)

        actual_prediction = _predict_ensemble(
            batch_trees[config_index], X_binned, config["learning_rate"]
        )
        np.testing.assert_allclose(actual_prediction, expected_prediction, rtol=1e-6, atol=1e-6)


def test_batch_recomputes_custom_loss_for_each_round_and_config():
    X, y = _regression_data()
    X_binned = ob.array(X, n_bins=254)
    configs = ob.ConfigBatch.from_grid(max_depth=[1, 2], n_rounds=3)
    calls = []

    def objective(prediction, target):
        calls.append(prediction.copy())
        return ob.mse_gradient(prediction, target)

    ob.fit_trees_batch(X_binned, configs=configs, y=y, loss=objective)

    assert len(calls) == configs.n_configs * configs.n_rounds
    assert any(np.any(prediction != 0) for prediction in calls[1:])


def test_multi_round_legacy_api_requires_targets():
    X, y = _regression_data()
    X_binned = ob.array(X, n_bins=254)
    configs = ob.ConfigBatch.from_grid(n_rounds=2)
    grad, hess = ob.mse_gradient(np.zeros_like(y), y)

    with pytest.raises(ValueError, match="y is required"):
        ob.fit_trees_batch(X_binned, grad, hess, configs)


def test_single_round_legacy_api_remains_supported():
    X, y = _regression_data()
    X_binned = ob.array(X, n_bins=254)
    configs = ob.ConfigBatch.from_grid(n_rounds=1)
    grad, hess = ob.mse_gradient(np.zeros_like(y), y)

    trees = ob.fit_trees_batch(X_binned, grad, hess, configs)

    assert len(trees) == 1
    assert len(trees[0]) == 1


def test_batch_validates_targets_and_configuration_count():
    X, y = _regression_data()
    X_binned = ob.array(X, n_bins=254)
    configs = ob.ConfigBatch.from_grid(n_rounds=2)

    with pytest.raises(ValueError, match="y must have shape"):
        ob.fit_trees_batch(X_binned, configs=configs, y=y[:-1])

    with pytest.raises(ValueError, match="At least one configuration"):
        ob.ConfigBatch.from_lists([], [], [], [])
