"""Fixed-power Tweedie mean geometry and round composition."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import MixedData, Problem, RunContext
from openboost.objectives import Tweedie
from openboost.outputs import positive_mean
from openboost.recipes import tweedie
from tests.v1.reference.mixed import Transformer, grow
from tests.v1.reference.positive import tweedie as reference


def fixture():
    x = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"], [None, "b"]],
        [0, 1, 2, 3, 4, 5],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    return Problem(
        x,
        [[0], [0.5], [4], [2], [0], [8]],
        x.row_ids,
        weight=[0.2, 2, 3, 0, 0.5, 1],
        offset=np.arange(6.0)[:, None] / 10,
    )


@pytest.mark.parametrize("power", [1.1, 1.5, 1.9])
def test_geometry_intercept_and_three_rounds(power):
    p = fixture()
    obj = Tweedie(power)
    w, y, offset = p.weight, p.target[:, 0], p.offset[:, 0]
    base = np.log(
        np.sum(w * y * np.exp((1 - power) * offset)) / np.sum(w * np.exp((2 - power) * offset))
    )
    np.testing.assert_allclose(obj.base(p), [base])
    raw = np.full((6, 1), base)
    np.testing.assert_allclose(np.dot(w, obj.geometry(p, raw)[1]), 0, atol=1e-12)
    fit = tweedie(p, p, context=RunContext("tweedie", 7), power=power, rounds=3, bins=4)
    transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    for step in fit.steps:
        loss, g, h = reference(p.with_offset(raw)[:, 0], y, power=power, weight=w)
        np.testing.assert_allclose(step.gradient, g)
        np.testing.assert_allclose(step.curvature, h)
        np.testing.assert_allclose(step.loss_before, loss)
        tree = grow(p.data.values, g[:, None], h[:, None], transform, weight=w)
        np.testing.assert_allclose(step.raw_before, raw)
        raw += 0.1 * tree.predict(p.data.values)
        np.testing.assert_allclose(step.raw_after, raw)
    assert fit.state.version == 3


def test_derivatives_include_zero_targets():
    p = fixture()
    obj = Tweedie()
    raw = np.arange(6.0)[:, None] / 4
    loss, g, h = obj.geometry(p, raw)
    epsilon = 1e-4
    for i in [0, 1, 4]:
        delta = np.zeros_like(raw)
        delta[i] = epsilon
        plus, minus = obj.loss(p, raw + delta), obj.loss(p, raw - delta)
        np.testing.assert_allclose(
            (plus - minus) / (2 * epsilon), g[i] * p.weight[i] / p.weight.sum(), atol=1e-8
        )
        np.testing.assert_allclose(
            (plus + minus - 2 * loss) / epsilon**2, h[i] * p.weight[i] / p.weight.sum(), atol=1e-6
        )


def test_all_zero_initializer_and_annualized_units():
    p = replace(fixture(), offset=np.zeros((6, 1)))
    zero = replace(p, target=np.zeros((6, 1)))
    assert Tweedie(minimum_mean=1e-4).base(zero)[0] == np.log(1e-4)
    fit = tweedie(zero, zero, context=RunContext("zero", 1), minimum_mean=1e-4)
    assert np.isfinite(fit.state.train_raw).all()
    # Period amounts become annualized targets; exposure enters weights only.
    exposure = np.array([0.5, 1, 2, 3, 0.2, 1])
    total = np.array([0, 2, 4, 6, 0, 3])
    annual = replace(p, target=(total / exposure)[:, None], weight=exposure)
    raw = np.zeros((6, 1))
    got = Tweedie().geometry(annual, raw)
    expected = reference(raw[:, 0], total / exposure, weight=exposure)
    for a, b in zip(got, expected, strict=True):
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(exposure * positive_mean(raw), exposure)


@pytest.mark.parametrize("power", [1, 2, np.nan])
def test_invalid_power(power):
    with pytest.raises(ValueError):
        Tweedie(power)


def test_invalid_support_structure_and_overflow():
    p = fixture()
    for invalid in (
        replace(p, target=np.full((6, 1), -1)),
        replace(p, structure={"exposure": np.ones((6, 1))}),
    ):
        with pytest.raises(ValueError):
            tweedie(invalid, invalid, context=RunContext("invalid", 1))
    with pytest.raises((ValueError, FloatingPointError)):
        Tweedie().geometry(p, np.full((6, 1), 2000))
    with pytest.raises(ValueError):
        Tweedie(minimum_mean=0)


def test_rejected_update_and_fresh_process_mean(tmp_path):
    import json
    import subprocess
    import sys

    from openboost.artifacts import Model
    from openboost.tree import depthwise

    p = fixture()

    def zero(data, fields):
        return depthwise(data, fields, max_depth=0, leaf=lambda *_: 0)

    rejected = tweedie(
        p, p, context=RunContext("reject", 1), learner=zero, step="backtracking", rounds=2
    )
    assert rejected.state.version == 0
    assert all(not s.accepted and s.raw_before is s.raw_after for s in rejected.steps)
    model = tweedie(p, p, context=RunContext("persist", 1)).state.model
    path = tmp_path / "model.json"
    model.save(path)
    assert Model.load(path).identity == model.identity
    code = """import json, sys
from openboost import MixedData
from openboost.artifacts import Model
from openboost.outputs import positive_mean
x = MixedData([[1,'unknown'], [None,None]], [10,11], ('x','c'), ('numeric','categorical'))
print(json.dumps(positive_mean(Model.load(sys.argv[1]).predict(x, offset=[[0.1],[0.2]])).tolist()))
"""
    x = MixedData(
        [[1, "unknown"], [None, None]], [10, 11], p.data.feature_names, p.data.feature_kinds
    )
    got = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    np.testing.assert_array_equal(got, positive_mean(model.predict(x, offset=[[0.1], [0.2]])))
