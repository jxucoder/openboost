"""Hand calculations for the v1 scalar oracle, independent of production."""

import numpy as np
import pytest

from tests.v1.reference.scalar import (
    newton_leaf,
    node_score,
    row_statistics,
    squared_error,
    sum_rows,
    weighted_mean,
)


def test_weighted_base_loss_and_unweighted_derivatives():
    y, weight = np.array([0.0, 2.0, 100.0]), np.array([1.0, 3.0, 0.0])
    assert weighted_mean(y, weight) == 1.5
    loss, gradient, curvature = squared_error(np.zeros(3), y, weight)
    assert loss == 1.5
    np.testing.assert_array_equal(gradient, [0.0, -2.0, -100.0])
    np.testing.assert_array_equal(curvature, np.ones(3))
    np.testing.assert_array_equal(
        row_statistics(gradient, curvature, weight), [[0.0, 1.0], [-6.0, 3.0], [0.0, 0.0]]
    )


def test_newton_leaf_minimizes_independent_quadratic():
    # Q(v)=sum_i w_i * (g_i*v + h_i*v*v/2) + lambda*v*v/2.
    g, h, w = [2.0, -1.0, 8.0], [1.0, 2.0, 0.5], [3.0, 2.0, 0.0]
    value = newton_leaf(4.0, 7.0, reg_lambda=1.0)
    assert value == -0.5

    def quadratic(v):
        return (
            sum(wi * (gi * v + hi * v * v / 2) for gi, hi, wi in zip(g, h, w, strict=True))
            + v * v / 2
        )

    assert node_score(4.0, 7.0, reg_lambda=1.0) == -quadratic(value) == 1.0
    for alternative in [-2.0, -0.51, -0.49, 0.0, 2.0]:
        assert quadratic(alternative) > quadratic(value)


def test_row_reduction_preserves_empty_and_zero_weight_rows():
    fields = row_statistics([2.0, -2.0, 100.0], [1.0, 1.0, 1.0], [1.0, 2.0, 0.0])
    np.testing.assert_array_equal(sum_rows(fields, (0, 1, 2)), [-2.0, 3.0])
    np.testing.assert_array_equal(sum_rows(fields, (2,)), [0.0, 0.0])
    np.testing.assert_array_equal(sum_rows(fields, ()), [0.0, 0.0])


@pytest.mark.parametrize("weight", [[0.0, 0.0], [-1.0, 2.0], [np.nan, 1.0], [np.inf, 1.0], [1.0]])
def test_invalid_training_weights_rejected(weight):
    with pytest.raises(ValueError, match="weight"):
        weighted_mean([1.0, 2.0], weight)


@pytest.mark.parametrize(
    "g,h,regularization",
    [
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, -1.0, 2.0),
        (np.nan, 1.0, 1.0),
        (1.0, np.inf, 1.0),
        (1.0, 1.0, -1.0),
    ],
)
def test_invalid_newton_state_rejected(g, h, regularization):
    with pytest.raises(ValueError):
        newton_leaf(g, h, reg_lambda=regularization)


@pytest.mark.parametrize(
    "raw,y", [([np.nan], [1.0]), ([1.0], [np.inf]), ([[1.0]], [1.0]), ([1.0, 2.0], [1.0])]
)
def test_invalid_objective_shape_or_values_rejected(raw, y):
    with pytest.raises(ValueError):
        squared_error(raw, y)


@pytest.mark.parametrize("rows", [(0, 0), (-1,), (2,), (0.5,)])
def test_invalid_row_selection_rejected(rows):
    with pytest.raises(ValueError, match="rows"):
        sum_rows(np.ones((2, 2)), rows)
