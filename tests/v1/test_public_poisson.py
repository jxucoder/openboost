"""Poisson count/exposure mechanics compared with independent formulas and trees."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import MixedData, Problem, RunContext
from openboost.objectives import Poisson
from openboost.outputs import poisson_mean
from openboost.recipes import poisson
from tests.v1.reference.mixed import Transformer, grow
from tests.v1.reference.positive import poisson as reference
from tests.v1.reference.positive import poisson_base


def fixture():
    x = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"], [None, "b"]],
        [0, 1, 2, 3, 4, 5],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    return Problem(
        x,
        [[0], [1], [4], [2], [0], [8]],
        x.row_ids,
        weight=[1, 2, 3, 0, 2, 1],
        offset=np.arange(6.0)[:, None] / 10,
        structure={"exposure": [[0.2], [0.5], [1], [2], [3], [0.8]]},
    )


def test_geometry_initialization_and_finite_differences():
    p = fixture()
    obj = Poisson()
    e = p.structure["exposure"][:, 0]
    base = obj.base(p)[0]
    expected = poisson_base(
        p.target[:, 0], e * np.exp(p.offset[:, 0]), weight=p.weight, minimum_rate=1e-6
    )
    np.testing.assert_allclose(base, expected)
    raw = np.full((6, 1), base)
    got = obj.geometry(p, raw)
    ref = reference(p.with_offset(raw)[:, 0], p.target[:, 0], e, weight=p.weight)
    for a, b in zip(got, ref, strict=True):
        np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.dot(p.weight, got[1]), 0, atol=1e-12)
    eps = 1e-4
    for i in [0, 1, 2]:
        delta = np.zeros((6, 1))
        delta[i] = eps
        plus, minus = obj.loss(p, raw + delta), obj.loss(p, raw - delta)
        np.testing.assert_allclose(
            (plus - minus) / (2 * eps), got[1][i] * p.weight[i] / p.weight.sum(), atol=1e-8
        )
        np.testing.assert_allclose(
            (plus + minus - 2 * got[0]) / eps**2,
            got[2][i] * p.weight[i] / p.weight.sum(),
            atol=1e-6,
        )


def test_three_rounds_match_independent_tree_and_geometry():
    p = fixture()
    fit = poisson(p, p, context=RunContext("poisson", 7), rounds=3, bins=4)
    transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    raw = np.full(
        (6, 1),
        poisson_base(
            p.target[:, 0],
            p.structure["exposure"][:, 0] * np.exp(p.offset[:, 0]),
            weight=p.weight,
            minimum_rate=1e-6,
        ),
    )
    for step in fit.steps:
        loss, g, h = reference(
            p.with_offset(raw)[:, 0], p.target[:, 0], p.structure["exposure"][:, 0], weight=p.weight
        )
        np.testing.assert_allclose(step.gradient, g)
        np.testing.assert_allclose(step.curvature, h)
        np.testing.assert_allclose(step.loss_before, loss)
        tree = grow(p.data.values, g[:, None], h[:, None], transform, weight=p.weight)
        np.testing.assert_allclose(step.raw_before, raw)
        raw += 0.1 * tree.predict(p.data.values)
        np.testing.assert_allclose(step.raw_after, raw)
    assert fit.state.version == 3


def test_zero_counts_and_exposure_scaling():
    p = fixture()
    zero = replace(p, target=np.zeros((6, 1)))
    obj = Poisson(minimum_rate=1e-4)
    assert obj.base(zero) == [np.log(1e-4)]
    fit = poisson(zero, zero, context=RunContext("zero", 1), minimum_rate=1e-4)
    assert np.isfinite(fit.state.train_raw).all()
    raw = np.arange(6.0)[:, None] / 5
    e = p.structure["exposure"][:, 0]
    one, two = poisson_mean(raw, e), poisson_mean(raw, 2 * e)
    np.testing.assert_allclose(two["rate"], one["rate"])
    np.testing.assert_allclose(two["count_mean"], 2 * one["count_mean"])


@pytest.mark.parametrize("kind", ["fraction", "negative", "zero_exposure", "missing", "extra"])
def test_reject_invalid_support_and_roles(kind):
    p = fixture()
    if kind in ("fraction", "negative"):
        p = replace(p, target=np.full((6, 1), 0.5 if kind == "fraction" else -1))
    else:
        structure = dict(p.structure)
        if kind == "zero_exposure":
            structure["exposure"] = np.zeros((6, 1))
        elif kind == "missing":
            structure = {}
        else:
            structure["ignored"] = np.ones((6, 1))
        p = replace(p, structure=structure)
    with pytest.raises(ValueError):
        poisson(p, p, context=RunContext("invalid", 1))


def test_overflow_and_backtracking_rejection_are_explicit():
    p = fixture()
    with pytest.raises((ValueError, FloatingPointError)):
        Poisson().geometry(p, np.full((6, 1), 1000))
    with pytest.raises(ValueError):
        poisson_mean(np.full((6, 1), -1000), np.ones(6))
    from openboost.tree import depthwise

    def zero(data, fields):
        return depthwise(data, fields, max_depth=0, leaf=lambda *_: 0)

    fit = poisson(
        p, p, context=RunContext("reject", 1), learner=zero, step="backtracking", rounds=2
    )
    assert fit.state.version == 0
    assert all(not s.accepted and s.raw_before is s.raw_after for s in fit.steps)


def test_fresh_process_count_rate_roundtrip(tmp_path):
    import json
    import subprocess
    import sys

    from openboost.artifacts import Model

    p = fixture()
    model = poisson(p, p, context=RunContext("persist", 1)).state.model
    path = tmp_path / "model.json"
    model.save(path)
    assert Model.load(path).identity == model.identity
    code = """import json, sys
from openboost import MixedData
from openboost.artifacts import Model
from openboost.outputs import poisson_mean
x = MixedData([[1, 'unknown'], [None, None]], [10,11], ('x','c'), ('numeric','categorical'))
raw = Model.load(sys.argv[1]).predict(x, offset=[[0.1],[0.2]])
print(json.dumps({k: v.tolist() for k,v in poisson_mean(raw, [0.5,2]).items()}))
"""
    x = MixedData(
        [[1, "unknown"], [None, None]], [10, 11], p.data.feature_names, p.data.feature_kinds
    )
    expected = poisson_mean(model.predict(x, offset=[[0.1], [0.2]]), [0.5, 2])
    got = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    for key in expected:
        np.testing.assert_array_equal(got[key], expected[key])
