"""Positive Gamma mean regression against independent formulas and tree growth."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import MixedData, Problem, RunContext
from openboost.objectives import Gamma
from openboost.outputs import positive_mean
from openboost.recipes import gamma
from tests.v1.reference.mixed import Transformer, grow
from tests.v1.reference.positive import gamma as reference
from tests.v1.reference.positive import gamma_base


def fixture():
    x = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"], [None, "b"]],
        [0, 1, 2, 3, 4, 5],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    return Problem(
        x,
        [[0.2], [1], [4], [2], [3], [8]],
        x.row_ids,
        weight=[1, 2, 3, 0, 2, 1],
        offset=np.arange(6.0)[:, None] / 10,
    )


def test_geometry_base_and_derivatives():
    p = fixture()
    base = Gamma.base(p)[0]
    np.testing.assert_allclose(
        base, gamma_base(p.target[:, 0] * np.exp(-p.offset[:, 0]), weight=p.weight)
    )
    raw = np.full((6, 1), base)
    got = Gamma.geometry(p, raw)
    expected = reference(p.with_offset(raw)[:, 0], p.target[:, 0], weight=p.weight)
    for a, b in zip(got, expected, strict=True):
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.dot(p.weight, got[1]), 0, atol=1e-12)
    eps = 1e-4
    for i in [0, 1, 2]:
        delta = np.zeros_like(raw)
        delta[i] = eps
        plus, minus = Gamma.loss(p, raw + delta), Gamma.loss(p, raw - delta)
        np.testing.assert_allclose(
            (plus - minus) / (2 * eps), got[1][i] * p.weight[i] / p.weight.sum(), atol=1e-8
        )
        np.testing.assert_allclose(
            (plus + minus - 2 * got[0]) / eps**2,
            got[2][i] * p.weight[i] / p.weight.sum(),
            atol=1e-6,
        )


def test_three_rounds_match_independent_tree():
    p = fixture()
    fit = gamma(p, p, context=RunContext("gamma", 7), rounds=3, bins=4)
    transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    raw = np.full((6, 1), gamma_base(p.target[:, 0] * np.exp(-p.offset[:, 0]), weight=p.weight))
    for step in fit.steps:
        loss, g, h = reference(p.with_offset(raw)[:, 0], p.target[:, 0], weight=p.weight)
        np.testing.assert_allclose(step.gradient, g)
        np.testing.assert_allclose(step.curvature, h)
        np.testing.assert_allclose(step.loss_before, loss)
        tree = grow(p.data.values, g[:, None], h[:, None], transform, weight=p.weight)
        np.testing.assert_allclose(step.raw_before, raw)
        raw += 0.1 * tree.predict(p.data.values)
        np.testing.assert_allclose(step.raw_after, raw)
    assert fit.state.version == 3


def test_claim_average_with_count_weights_matches_claim_level_geometry():
    # Equal predictors within each policy: aggregate claim geometry exactly.
    x = MixedData([[0, "a"], [1, "b"]], [0, 1], ("x", "c"), ("numeric", "categorical"))
    p = Problem(x, [[3], [5]], x.row_ids, weight=[2, 3])
    raw = np.array([[0.2], [0.7]])
    loss, g, h = Gamma.geometry(p, raw)
    expected = reference([0.2, 0.2, 0.7, 0.7, 0.7], [2, 4, 3, 5, 7])
    np.testing.assert_allclose(loss, expected[0])
    for got, claim in zip((g, h), expected[1:], strict=True):
        np.testing.assert_allclose(got * p.weight, [sum(claim[:2]), sum(claim[2:])])
    # Unit policy weights deliberately produce a different aggregate objective.
    assert Gamma.loss(replace(p, weight=[1, 1]), raw) != loss


@pytest.mark.parametrize("kind", ["zero", "negative", "structure", "infinite_raw"])
def test_invalid_inputs_fail(kind):
    p = fixture()
    if kind in ("zero", "negative"):
        p = replace(p, target=np.full((6, 1), 0 if kind == "zero" else -1))
    elif kind == "structure":
        p = replace(p, structure={"exposure": np.ones((6, 1))})
    if kind == "infinite_raw":
        with pytest.raises((ValueError, FloatingPointError)):
            Gamma.geometry(p, np.full((6, 1), -1000))
    else:
        with pytest.raises(ValueError):
            gamma(p, p, context=RunContext("invalid", 1))


def test_backtracking_rejection_and_positive_output_support():
    from openboost.tree import depthwise

    p = fixture()

    def zero(data, fields):
        return depthwise(data, fields, max_depth=0, leaf=lambda *_: 0)

    fit = gamma(p, p, context=RunContext("reject", 1), learner=zero, step="backtracking", rounds=2)
    assert fit.state.version == 0
    assert all(not s.accepted and s.raw_before is s.raw_after for s in fit.steps)
    for raw in ([[1000]], [[-1000]], [[1, 2]]):
        with pytest.raises((ValueError, FloatingPointError)):
            positive_mean(raw)


def test_fresh_process_mean_roundtrip(tmp_path):
    import json
    import subprocess
    import sys

    from openboost.artifacts import Model

    p = fixture()
    model = gamma(p, p, context=RunContext("persist", 1)).state.model
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
    expected = positive_mean(model.predict(x, offset=[[0.1], [0.2]]))
    got = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    np.testing.assert_array_equal(got, expected)
