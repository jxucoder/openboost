"""Classification oracles checked against calculus and independent updates."""

import numpy as np
import pytest

from .reference.classification import ClassMap, binary, binary_base, softmax
from .reference.scalar import row_statistics
from .reference.tree import fit_tree


def test_class_mapping_and_invalid_targets():
    schema = ClassMap.fit(["yes", "no", "yes"])
    assert schema.values == ("no", "yes")
    assert schema.encode(["yes", "no"]) == (1, 0)
    assert schema.decode([1, 0]) == ("yes", "no")
    for labels in (["only"], ["a", None], [1, "1"]):
        with pytest.raises(ValueError):
            ClassMap.fit(labels)
    with pytest.raises(ValueError):
        schema.encode(["unknown"])
    with pytest.raises(ValueError):
        schema.decode([2])
    with pytest.raises(ValueError):
        schema.decode([0.5])


def test_binary_hand_values_weights_and_extremes():
    loss, g, h = binary([0, 0], [0, 1], weight=[1, 3])
    assert loss == pytest.approx(np.log(2))
    np.testing.assert_allclose(g, [0.5, -0.5])
    np.testing.assert_allclose(h, [0.25, 0.25])
    np.testing.assert_allclose(row_statistics(g, h, [1, 3]), [[0.5, 0.25], [-1.5, 0.75]])
    assert binary_base([0, 1], weight=[1, 3], clip=1e-6) == pytest.approx(np.log(3))
    assert binary_base([0, 1], weight=[0, 1], clip=0.01) == pytest.approx(np.log(99))
    extreme, g, h = binary([-1000, 1000], [1, 0])
    assert extreme == 1000
    np.testing.assert_array_equal(g, [-1, 1])
    np.testing.assert_array_equal(h, [0, 0])
    assert binary([100, -100], [1, 0])[0] > 0  # no cancellation to zero


@pytest.mark.parametrize("target", [[0, 2], [0, 0.5], [0, np.nan]])
def test_binary_rejects_invalid_codes(target):
    with pytest.raises(ValueError):
        binary([0, 0], target)


@pytest.mark.parametrize("clip", [0, 0.5, np.nan, 1e-20])
def test_binary_base_requires_explicit_valid_clipping(clip):
    with pytest.raises(ValueError):
        binary_base([0, 1], clip=clip)


def test_softmax_exact_hessian_and_named_diagonal_bound():
    loss, p, g, exact, bound = softmax([[0, 0, 0]], [1])
    assert loss == pytest.approx(np.log(3))
    np.testing.assert_allclose(p, [[1 / 3] * 3])
    np.testing.assert_allclose(g, [[1 / 3, -2 / 3, 1 / 3]])
    np.testing.assert_allclose(exact[0], np.eye(3) / 3 - np.ones((3, 3)) / 9)
    np.testing.assert_allclose(bound, [[4 / 9] * 3])
    assert np.linalg.eigvalsh(np.diag(bound[0]) - exact[0]).min() >= -1e-14


def test_softmax_finite_differences_shift_and_class_permutation():
    raw = np.array([[0.3, -1.2, 2.1]])
    loss, p, g, exact, bound = softmax(raw, [2])
    eps = 1e-5
    for k in range(3):
        delta = np.eye(3)[k : k + 1] * eps
        plus = softmax(raw + delta, [2])
        minus = softmax(raw - delta, [2])
        assert (plus[0] - minus[0]) / (2 * eps) == pytest.approx(g[0, k], abs=1e-9)
        np.testing.assert_allclose((plus[2] - minus[2]) / (2 * eps), exact[:, :, k], atol=1e-10)
    perm = [2, 0, 1]
    shifted = softmax(raw[:, perm] + 1000, [0])
    assert shifted[0] == pytest.approx(loss)
    np.testing.assert_allclose(shifted[1], p[:, perm])
    np.testing.assert_allclose(shifted[2], g[:, perm])
    np.testing.assert_allclose(shifted[4], bound[:, perm])
    assert np.sum(p) == pytest.approx(1)
    assert np.sum(g) == pytest.approx(0, abs=1e-15)


def test_softmax_extremes_weights_and_invalid_labels():
    raw = [[1000, -1000, 0], [0, 0, 0]]
    loss, _, g, exact, bound = softmax(raw, [1, 0], weight=[1, 3])
    assert loss == pytest.approx((2000 + 3 * np.log(3)) / 4)
    unweighted = softmax(raw, [1, 0])
    for actual, expected in zip((g, exact, bound), unweighted[2:], strict=True):
        np.testing.assert_array_equal(actual, expected)
    for labels in ([3, 0], [-1, 0], [0.5, 0], [np.nan, 0]):
        with pytest.raises(ValueError):
            softmax(raw, labels)


def test_two_binary_rounds_recompute_derivatives_and_validation_predictions():
    bins = [[0], [0], [1], [1]]
    target = [0, 0, 1, 1]
    raw = np.zeros(4)
    prediction = np.zeros(2)
    for step in range(2):
        _, g, h = binary(raw, target)
        tree = fit_tree(bins, g, h, max_depth=1)
        if step == 0:
            np.testing.assert_allclose(tree.predict([[0], [1]]), [-2 / 3, 2 / 3])
        else:
            p = 1 / (1 + np.exp(1 / 15))
            value = 2 * p / (1 + 2 * p * (1 - p))
            np.testing.assert_allclose(tree.predict([[0], [1]]), [-value, value])
        raw += 0.1 * tree.predict(bins)
        prediction += 0.1 * tree.predict([[0], [1]])
    np.testing.assert_allclose(raw, prediction[[0, 0, 1, 1]])
    assert binary(raw, target)[0] < np.log(2)


def test_two_softmax_rounds_use_joint_snapshot_and_permutation_equivariance():
    bins = [[0], [1], [2]]
    raw = np.zeros((3, 3))
    perm = [2, 0, 1]
    reordered = raw[:, perm].copy()
    previous = None
    for _ in range(2):
        _, _, g, _, bound = softmax(raw, [0, 1, 2])
        _, _, gp, _, hp = softmax(reordered, [1, 2, 0])
        if previous is not None:
            assert not np.allclose(g, previous)
        updates = np.column_stack(
            [fit_tree(bins, g[:, k], bound[:, k]).predict(bins) for k in range(3)]
        )
        updates_p = np.column_stack(
            [fit_tree(bins, gp[:, k], hp[:, k]).predict(bins) for k in range(3)]
        )
        raw = raw + 0.1 * updates
        reordered = reordered + 0.1 * updates_p
        np.testing.assert_allclose(reordered, raw[:, perm], atol=1e-15)
        previous = g.copy()
    assert softmax(raw, [0, 1, 2])[0] < np.log(3)


def test_binary_derivatives_by_finite_difference_and_weight_replication():
    raw = np.array([-0.7, 1.3])
    y = [0, 1]
    loss, g, h = binary(raw, y, weight=[1, 3])
    repeated = binary(raw[[0, 1, 1, 1]], [0, 1, 1, 1])
    assert repeated[0] == pytest.approx(loss)
    eps = 1e-5
    for i in range(2):
        plus = binary([raw[i] + eps], [y[i]])
        minus = binary([raw[i] - eps], [y[i]])
        assert (plus[0] - minus[0]) / (2 * eps) == pytest.approx(g[i], abs=1e-10)
        assert (plus[1][0] - minus[1][0]) / (2 * eps) == pytest.approx(h[i], abs=1e-10)


def test_softmax_two_round_root_updates_have_independent_hand_solution():
    # All channels use one joint snapshot. Unequal class masses make a root
    # update nonzero and expose both weighting and stale second-round geometry.
    bins = [[0], [0], [0]]
    target = [0, 1, 2]
    weight = np.array([1.0, 2.0, 3.0])
    raw = np.zeros((3, 3))
    validation = np.zeros((1, 3))
    for step in range(2):
        _, _, g, _, bound = softmax(raw, target, weight=weight)
        update = np.array(
            [
                fit_tree(bins, g[:, k], bound[:, k], weight=weight, max_depth=0).predict([[0]])[0]
                for k in range(3)
            ]
        )
        if step == 0:
            expected = np.array([-3 / 11, 0, 3 / 11])
        else:
            exp = np.exp(np.array([-3 / 110, 0, 3 / 110]))
            p = exp / sum(exp)
            expected = (weight - 6 * p) / (1 + 12 * p * (1 - p))
        np.testing.assert_allclose(update, expected, atol=1e-15)
        raw += 0.1 * update
        validation += 0.1 * update
    np.testing.assert_allclose(raw, np.repeat(validation, 3, axis=0))


@pytest.mark.parametrize(
    "raw,target",
    [([], []), ([[0]], [0]), ([[0, np.inf]], [0]), ([[0, 0]], []), ([[1e308, -1e308]], [0])],
)
def test_softmax_invalid_shapes_and_unrepresentable_range(raw, target):
    with pytest.raises(ValueError):
        softmax(raw, target)


def test_single_class_base_rejected_and_zero_weight_rows_excluded_from_loss():
    with pytest.raises(ValueError):
        binary_base([1, 1], clip=1e-6)
    assert binary([0, -1000], [1, 1], weight=[1, 0])[0] == pytest.approx(np.log(2))
    assert softmax([[0, 0, 0], [-1000, 0, 1000]], [0, 0], weight=[1, 0])[0] == pytest.approx(
        np.log(3)
    )
