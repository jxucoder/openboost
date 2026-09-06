"""Normal distributional recipe against independent geometry and update oracles."""

import numpy as np

from openboost import NumericData, Problem


def test_raw_width_is_independent_of_target_width():
    x = NumericData([[0], [1]], [1, 2], ("x",))
    p = Problem(x, [[2], [3]], x.row_ids, raw_width=2)
    assert p.target.shape == (2, 1) and p.offset.shape == (2, 2)
    np.testing.assert_array_equal(p.with_offset([[1, 2], [3, 4]]), [[1, 2], [3, 4]])


import json
import subprocess
import sys

import pytest

from openboost import RunContext
from openboost.artifacts import Model
from openboost.binning import NumericBinning
from openboost.objectives import Normal, Squared, diagonal_direction
from openboost.recipes import normal
from openboost.stats import least_squares
from openboost.tree import depthwise
from tests.v1.reference.coupled import directions, normal_base, normal_scores, step
from tests.v1.reference.coupled import normal as reference_normal


def fixture():
    x = NumericData([[0], [1], [2], [3], [4], [np.nan]], np.arange(6), ("x",))
    p = Problem(
        x, [[-4], [-1], [0], [1], [3], [8]], x.row_ids, weight=[1, 0, 2, 1, 3, 2], raw_width=2
    )
    return p


@pytest.mark.parametrize("mode", ["ordinary", "natural"])
def test_geometry_base_and_three_rounds_match_reference(mode):
    p = fixture()
    result = normal(p, p, context=RunContext(mode, 3), rounds=3, bins=6, mode=mode)
    base = normal_base(p.target[:, 0], minimum_scale=1e-6, weight=p.weight)
    np.testing.assert_allclose(result.state.model.base, base)
    raw = np.broadcast_to(base, (6, 2)).copy()
    b = NumericBinning.fit(p.data, bins=6).transform(p.data)
    bins = np.where(b.missing.T, np.nan, b.codes.T)
    for actual in result.steps:
        loss, gradient, metric = reference_normal(raw, p.target[:, 0], weight=p.weight)
        expected_direction = directions(
            gradient, metric, mode="full" if mode == "natural" else mode
        )
        np.testing.assert_allclose(actual.gradient, gradient)
        np.testing.assert_allclose(actual.fisher_diagonal, np.diagonal(metric, axis1=1, axis2=2))
        np.testing.assert_allclose(actual.direction, expected_direction)
        ref = step(
            bins,
            raw,
            p.target[:, 0],
            reference_normal,
            weight=p.weight,
            mode="full" if mode == "natural" else mode,
            rates=tuple(0.1 * 0.5**j for j in range(6)),
        )
        assert actual.accepted == ref.accepted
        assert actual.coefficients == tuple(t[0] for t in ref.trials)
        np.testing.assert_allclose(actual.raw_before, raw)
        np.testing.assert_allclose(actual.raw_after, ref.raw_after)
        np.testing.assert_allclose([actual.loss_before, actual.loss_after], [loss, ref.loss_after])
        raw = np.asarray(ref.raw_after)
    nll, crps = normal_scores(raw, p.target[:, 0], weight=p.weight)
    assert Normal.loss(p, result.state.train_raw) == pytest.approx(nll)
    assert np.isfinite(crps)
    np.testing.assert_allclose(Normal.parameters(raw)[:, 1], np.exp(raw[:, 1]))


def test_offset_geometry_and_base_first_order_conditions():
    p = fixture()
    offsets = np.column_stack((np.arange(6) / 2, np.arange(6) / 10))
    p = Problem(p.data, p.target, p.row_ids, weight=p.weight, offset=offsets, raw_width=2)
    base = Normal.base(p)
    raw = np.broadcast_to(base, (6, 2))
    loss, g, h = Normal.geometry(p, raw)
    expected = reference_normal(raw + offsets, p.target[:, 0], weight=p.weight)
    np.testing.assert_allclose(g, expected[1])
    np.testing.assert_allclose(h, np.diagonal(expected[2], axis1=1, axis2=2))
    assert loss == pytest.approx(expected[0])
    np.testing.assert_allclose(np.sum(p.weight[:, None] * g, axis=0), [0, 0], atol=1e-12)
    result = normal(p, p, context=RunContext("offsets", 1), rounds=2)
    np.testing.assert_array_equal(
        result.state.model.predict(p.data, offset=offsets), result.state.train_raw + offsets
    )
    z = diagonal_direction(g, h)
    fields = least_squares(p, z[:, 0])
    np.testing.assert_allclose(fields.values[:, 0], -p.weight * z[:, 0])
    np.testing.assert_allclose(fields.values[:, 1], p.weight)


def test_joint_full_rejection_and_nonfinite_backtracking():
    p = fixture()
    calls = []

    def zero(b, fields):
        calls.append(1)
        return depthwise(b, fields, max_depth=0, leaf=lambda *_: 0)

    result = normal(p, p, context=RunContext("reject", 1), learner=zero, rounds=2)
    assert len(calls) == 4 and result.state.version == 0
    assert not result.state.model.terms and result.state.best_model is result.state.model
    for item in result.steps:
        assert not item.accepted and len(item.coefficients) == 6
        assert item.raw_after is item.raw_before
    # Nonfinite distribution trials must not corrupt accepted state.
    calls.clear()
    result = normal(p, p, context=RunContext("overflow", 1), rounds=1, learning_rate=1e5)
    assert any(result.steps[0].failures)
    assert not result.steps[0].accepted and result.state.version == 0
    assert np.isfinite(result.state.train_raw).all()


def test_normal_model_roundtrip_without_training_data(tmp_path):
    p = fixture()
    result = normal(p, p, context=RunContext("persist", 1), rounds=2, step="fixed")
    assert result.state.version == 2 and len(result.state.model.terms) == 4
    path = tmp_path / "normal.json"
    result.state.model.save(path)
    assert Model.load(path).identity == result.state.model.identity
    code = """import json, sys
from openboost import NumericData
from openboost.artifacts import Model
from openboost.objectives import Normal
x = NumericData([[-100], [100], [float('nan')]], [1, 2, 3], ('x',))
print(json.dumps(Normal.parameters(Model.load(sys.argv[1]).predict(x)).tolist()))
"""
    output = subprocess.check_output([sys.executable, "-c", code, str(path)], text=True)
    x = NumericData([[-100], [100], [np.nan]], [1, 2, 3], ("x",))
    np.testing.assert_allclose(json.loads(output), Normal.parameters(result.state.model.predict(x)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"raw_width": 0},
        {"raw_width": True},
        {"raw_width": 2, "offset": [[0], [0]]},
        {"raw_width": 1.5},
    ],
)
def test_invalid_raw_shapes_rejected(kwargs):
    x = NumericData([[0], [1]], [1, 2], ("x",))
    with pytest.raises(ValueError):
        Problem(x, [[1], [2]], x.row_ids, **kwargs)


def test_invalid_geometry_zero_rounds_and_scalar_recipe_boundary():
    p = fixture()
    with pytest.raises(ValueError):
        Squared.validate(p)
    for kwargs in (
        {"mode": "full"},
        {"damping": -1},
        {"minimum_scale": 0},
        {"mode": "ordinary", "damping": 1},
    ):
        with pytest.raises(ValueError):
            normal(p, p, context=RunContext("invalid", 1), rounds=0, **kwargs)
    with pytest.raises(ValueError):
        Normal.parameters([[0, -1000]])
    ordinary = Problem(p.data, p.target, p.row_ids)
    assert ordinary.identity != p.identity


def test_nonfinite_trials_continue_to_finite_trials_and_schema_errors_raise():
    p = fixture()
    channels = []

    def shrink_scale(b, fields):
        value = 0 if not channels else -10
        channels.append(1)
        return depthwise(b, fields, max_depth=0, leaf=lambda *_: value)

    result = normal(
        p,
        p,
        context=RunContext("finite-retry", 1),
        rounds=1,
        learner=shrink_scale,
        learning_rate=100,
    )
    failures = result.steps[0].failures
    assert len(channels) == 2 and failures[0] is not None and failures[-1] is None
    assert not result.steps[0].accepted and result.state.version == 0

    def wrong_schema(b, fields):
        from dataclasses import replace

        tree = depthwise(b, fields)
        return replace(tree, binning=NumericBinning(("wrong",), tree.binning.cuts))

    with pytest.raises(ValueError, match="schema"):
        normal(p, p, context=RunContext("schema", 1), rounds=1, learner=wrong_schema)
