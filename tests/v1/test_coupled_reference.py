"""A11/A12 geometry and accepted-state checks independent of production."""

import math

import numpy as np
import pytest

from .reference.coupled import (
    directions,
    formula,
    formula_base,
    formula_predict,
    normal,
    normal_base,
    normal_scores,
    step,
)


def test_normal_hand_fisher_and_natural_direction():
    raw = [[1, math.log(2)]]
    loss, gradient, fisher = normal(raw, [3])
    assert loss == pytest.approx(math.log(2) + 0.5 + 0.5 * math.log(2 * math.pi))
    np.testing.assert_allclose(gradient, [[-0.5, 0]])
    np.testing.assert_allclose(fisher, [[[0.25, 0], [0, 2]]])
    np.testing.assert_allclose(directions(gradient, fisher, mode="full"), [[2, 0]])
    np.testing.assert_allclose(directions(gradient, fisher, mode="ordinary"), [[0.5, 0]])
    base = normal_base([1, 3], weight=[1, 3], minimum_scale=0.1)
    np.testing.assert_allclose(base, [2.5, 0.5 * math.log(0.75)])
    np.testing.assert_allclose(normal_base([3, 3], minimum_scale=0.1), [3, math.log(0.1)])


@pytest.mark.parametrize("kind", ["normal", "formula"])
def test_geometry_gradients_and_formula_jacobian_by_finite_difference(kind):
    raw = np.array([[0.2, -0.3], [1.0, 0.4]])
    y, x = [2.0, 1.0], [1.0, 2.0]
    objective = normal if kind == "normal" else lambda r, t: formula(r, t, x)
    _, g, metric = objective(raw, y)
    eps = 1e-5
    for row in range(2):
        for channel in range(2):
            delta = np.zeros_like(raw)
            delta[row, channel] = eps
            assert (objective(raw + delta, y)[0] - objective(raw - delta, y)[0]) / (
                2 * eps
            ) * 2 == pytest.approx(g[row, channel], abs=1e-8)
    if kind == "formula":
        jacobian = np.column_stack(
            [
                (
                    formula_predict(raw + np.eye(2)[k] * eps, x)[0]
                    - formula_predict(raw - np.eye(2)[k] * eps, x)[0]
                )
                / (2 * eps)
                for k in range(2)
            ]
        )
        np.testing.assert_allclose(metric, np.einsum("ni,nj->nij", jacobian, jacobian), atol=1e-10)
        assert all(abs(np.linalg.det(m)) < 1e-14 for m in metric)


def test_formula_direction_solve_and_rank_deficiency():
    raw = np.zeros((1, 2))
    _, g, metric = formula(raw, [2], [1])
    damping = 0.2
    full = directions(g, metric, mode="full", damping=damping)
    np.testing.assert_allclose((metric[0] + damping * np.eye(2)) @ full[0], -g[0])
    diagonal = directions(g, metric, mode="diagonal", damping=damping)
    np.testing.assert_allclose(diagonal, -g / (np.diagonal(metric, axis1=1, axis2=2) + damping))
    assert not np.allclose(full, diagonal)
    with pytest.raises(ValueError):
        directions(g, metric, mode="full")


def test_formula_monotonicity_initialization_and_nonidentifiability():
    base = formula_base([2, 4], weight=[1, 3])
    prediction, a, b = formula_predict(np.tile(base, (3, 1)), [0.5, 1, 2])
    np.testing.assert_allclose(a, 3.5)
    np.testing.assert_allclose(b, 1)
    assert np.all(np.diff(prediction) > 0)

    # Distinct a,b fit the same one-age observation exactly.
    def inverse(v):
        return math.log(math.expm1(v))

    alternatives = [[inverse(2), inverse(math.log(2))], [inverse(3), inverse(-math.log(2 / 3))]]
    np.testing.assert_allclose(formula_predict(alternatives, [1, 1])[0], [1, 1])
    # A decreasing pair at the same recipe Z cannot fit this increasing family.
    predictions = formula_predict([base, base], [1, 2])[0]
    assert np.sum((predictions - [3, 1]) ** 2) >= 2


def test_normal_evaluator_independent_nll_and_crps():
    raw = [[0, 0], [1, math.log(2)]]
    nll, crps = normal_scores(raw, [0, 1], weight=[1, 3])
    expected = (math.sqrt(2) - 1) / math.sqrt(math.pi)
    assert crps == pytest.approx(expected * 7 / 4)
    assert nll == pytest.approx(0.5 * math.log(2 * math.pi) + 0.75 * math.log(2))


@pytest.mark.parametrize("mode", ["ordinary", "diagonal", "full"])
def test_formula_two_round_joint_and_validation_reconstruction(mode):
    bins, x = [[0], [0], [1], [1]], [1.0, 2.0, 1.0, 2.0]
    truth = np.array([[2.0, 1.0], [2.0, 1.0], [3.0, 0.5], [3.0, 0.5]])
    raw_true = np.log(np.expm1(truth))
    y = formula_predict(raw_true, x)[0]
    raw = np.zeros((4, 2))
    reconstructed = raw.copy()
    previous = None

    def objective(r, t, weight=None):
        return formula(r, t, x, weight=weight)

    for _ in range(2):
        _, g, metric = objective(raw, y)
        result = step(bins, raw, y, objective, mode=mode, damping=0.1, rates=(1.0, 0.5, 0.1))
        assert result.accepted
        assert result.loss_after < result.loss_before
        if previous is not None:
            assert not np.allclose(g, previous)
        # Every tree targets the direction from the SAME snapshot.
        direction = directions(g, metric, mode=mode, damping=0.1)
        for channel, tree, coefficient in result.terms:
            for node in tree.nodes:
                if node.condition is None:
                    assert node.value == pytest.approx(
                        sum(direction[list(node.rows), channel]) / (1 + len(node.rows))
                    )
            reconstructed[:, channel] += coefficient * tree.predict(bins)
        raw = np.array(result.raw_after)
        np.testing.assert_allclose(reconstructed, raw)
        previous = g.copy()


def test_normal_two_round_natural_root_hand_updates_and_weights():
    y, raw = [1.0, 3.0], np.zeros((2, 2))
    for _ in range(2):
        mu, ell = raw[0]
        expected_mu = (7 - 3 * mu) / 4
        expected_scale = (0.5 * ((1 - mu) ** 2 + 2 * (3 - mu) ** 2) * math.exp(-2 * ell) - 1.5) / 4
        result = step(
            [[0], [0]],
            raw,
            y,
            normal,
            weight=[1, 2],
            mode="full",
            rates=(0.1,),
            require_decrease=False,
        )
        assert result.accepted
        np.testing.assert_allclose(
            np.array(result.raw_after)[0], raw[0] + 0.1 * np.array([expected_mu, expected_scale])
        )
        raw = np.array(result.raw_after)


def test_ordered_recomputes_geometry_and_differs_from_joint():
    bins, joint_raw, y = [[0], [0]], np.zeros((2, 2)), [1.0, 3.0]
    ordered_raw = joint_raw.copy()
    for _ in range(2):
        joint = step(bins, joint_raw, y, normal, mode="full", rates=(0.1,), require_decrease=False)
        first = step(
            bins,
            ordered_raw,
            y,
            normal,
            mode="full",
            channels=(0,),
            rates=(0.1,),
            require_decrease=False,
        )
        second = step(
            bins,
            first.raw_after,
            y,
            normal,
            mode="full",
            channels=(1,),
            rates=(0.1,),
            require_decrease=False,
        )
        assert not np.allclose(second.raw_after, joint.raw_after)
        np.testing.assert_allclose(
            np.array(second.raw_after)[:, 0], np.array(joint.raw_after)[:, 0]
        )
        joint_raw, ordered_raw = np.array(joint.raw_after), np.array(second.raw_after)


def test_rejected_candidate_preserves_snapshot_and_has_no_terms():
    raw = np.zeros((2, 2))
    original = raw.copy()
    result = step([[0], [0]], raw, [1.0, 3.0], normal, mode="full", rates=(1000.0,))
    assert not result.accepted and result.terms == ()
    assert result.raw_before == result.raw_after
    np.testing.assert_array_equal(raw, original)
    raw[:] = 8
    np.testing.assert_array_equal(result.raw_after, original)


def test_backtracking_records_rejections_and_commits_only_accepted_coefficient():
    result = step(
        [[0], [0]], np.zeros((2, 2)), [1.0, 3.0], normal, mode="full", rates=(1000.0, 5.0, 0.1)
    )
    assert result.accepted
    assert len(result.trials) == 3
    assert result.trials[0][1] is None
    assert result.trials[1][1] > result.loss_before
    assert all(coefficient == 0.1 for _, _, coefficient in result.terms)
    direct = step([[0], [0]], np.zeros((2, 2)), [1.0, 3.0], normal, mode="full", rates=(0.1,))
    np.testing.assert_allclose(result.raw_after, direct.raw_after)


def test_direction_fit_weight_replication_and_evaluator_agreement():
    raw = np.array([[0.1, -0.2], [0.3, 0.5]])
    y = np.array([1.0, 3.0])
    weighted = step([[0], [1]], raw, y, normal, weight=[1, 3], rates=(0.1,))
    ids = [0, 1, 1, 1]
    repeated = step([[0], [1], [1], [1]], raw[ids], y[ids], normal, rates=(0.1,))
    np.testing.assert_allclose(np.array(weighted.raw_after)[ids], repeated.raw_after)
    assert normal_scores(raw, y, weight=[1, 3])[0] == pytest.approx(
        normal(raw, y, weight=[1, 3])[0]
    )


@pytest.mark.parametrize("rates", [(), (0,), (-1,), (np.nan,)])
def test_invalid_step_coefficients(rates):
    with pytest.raises(ValueError):
        step([[0]], [[0, 0]], [1], normal, rates=rates)


@pytest.mark.parametrize("x", [[0], [-1], [np.nan], []])
def test_invalid_formula_structure(x):
    with pytest.raises(ValueError):
        formula([[0, 0]], [1], x)


def test_formula_small_positive_parameters_use_stable_expm1():
    prediction, a, b = formula_predict([[-30, -30]], [1e-6])
    assert prediction[0] > 0
    assert prediction[0] == pytest.approx(a[0] * b[0] * 1e-6, rel=1e-12, abs=0)


def test_bad_metric_damping_and_normal_initialization_rejected():
    with pytest.raises(ValueError):
        directions([[1, 1]], [[[1, 0], [0, 1]]], damping=-1)
    with pytest.raises(ValueError):
        directions([[1, 1]], [[[1, 2], [0, 1]]], mode="full")
    with pytest.raises(ValueError):
        normal_base([1, 1], minimum_scale=0)


def test_formula_rejects_underflowed_positive_parameters():
    with pytest.raises(ValueError):
        formula_predict([[-1000, 0]], [1])
