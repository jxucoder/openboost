"""Censored AFT geometry, tail quadrature and scale-aware persistence."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import MixedData, Problem, RunContext
from openboost.recipes import aft
from openboost.survival import AFTModel, LogNormalAFT, normal_tail
from tests.v1.reference.mixed import Transformer, grow
from tests.v1.reference.survival import aft as reference
from tests.v1.reference.survival import aft_predict
from tests.v1.reference.survival import normal_tail as reference_tail


def fixture():
    x = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"], [None, "b"]],
        [0, 1, 2, 3, 4, 5],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    return Problem(
        x,
        [[1, 1], [2, np.inf], [3, 3], [5, np.inf], [8, 8], [10, np.inf]],
        x.row_ids,
        target_kind="event_right",
        weight=[1, 2, 3, 0, 1, 2],
        offset=np.arange(6.0)[:, None] / 10,
    )


@pytest.mark.parametrize("z", [-40, -5, 0, 5, 8, 8.01, 12, 40, 1000, 1e8])
def test_tail_matches_independent_continued_fraction(z):
    np.testing.assert_allclose(normal_tail(z), reference_tail(z), rtol=2e-12, atol=1e-14)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
def test_three_round_likelihood_and_trees(sigma):
    p = fixture()
    obj = LogNormalAFT(sigma)
    raw = np.full((6, 1), np.average(np.log(p.target[:, 0]) - p.offset[:, 0], weights=p.weight))
    fit = aft(p, p, context=RunContext("aft", 7), sigma=sigma, rounds=3, bins=4)
    transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    for step in fit.steps:
        loss, g, h = reference(
            p.with_offset(raw)[:, 0], p.target[:, 0], p.target[:, 1], sigma=sigma, weight=p.weight
        )
        np.testing.assert_allclose(step.gradient, g)
        np.testing.assert_allclose(step.curvature, h)
        np.testing.assert_allclose(step.loss_before, loss)
        tree = grow(p.data.values, g[:, None], h[:, None], transform, weight=p.weight)
        np.testing.assert_allclose(step.raw_before, raw)
        raw += 0.1 * tree.predict(p.data.values)
        np.testing.assert_allclose(step.raw_after, raw)
        np.testing.assert_allclose(step.loss_after, obj.loss(p, raw))
    assert fit.state.version == 3


def test_censoring_changes_geometry_and_finite_differences():
    p = fixture()
    obj = LogNormalAFT(0.7)
    raw = np.zeros((6, 1))
    loss, g, h = obj.geometry(p, raw)
    event = replace(p, target=np.column_stack((p.target[:, 0], p.target[:, 0])))
    assert event.identity != p.identity
    assert obj.loss(event, raw) != loss
    assert not np.allclose(obj.geometry(event, raw)[1], g)
    eps = 1e-4
    for i in [0, 1, 2]:
        delta = np.zeros_like(raw)
        delta[i] = eps
        plus, minus = obj.loss(p, raw + delta), obj.loss(p, raw - delta)
        np.testing.assert_allclose(
            (plus - minus) / (2 * eps), g[i] * p.weight[i] / p.weight.sum(), atol=1e-8
        )
        np.testing.assert_allclose(
            (plus + minus - 2 * loss) / eps**2, h[i] * p.weight[i] / p.weight.sum(), atol=1e-6
        )


@pytest.mark.parametrize(
    "bounds", [[[0, 1]], [[2, 1]], [[1, 2]], [[1, np.nan]], [[np.inf, np.inf]], [[1, -np.inf]]]
)
def test_invalid_or_unsupported_censoring(bounds):
    p = fixture()
    with pytest.raises(ValueError):
        Problem(p.data, np.repeat(bounds, 6, axis=0), p.row_ids, target_kind="event_right")


def test_explicit_target_kind_and_ordinary_target_rejection():
    p = fixture()
    assert p.raw_width == 1 and np.isposinf(p.target[1, 1])
    with pytest.raises(ValueError):
        replace(p, target_kind="numeric")
    with pytest.raises(ValueError):
        p.target.setflags(write=True)
    from openboost.recipes import squared

    with pytest.raises(ValueError):
        squared(p, p, context=RunContext("wrong", 1))
    for sigma in [0, -1, np.inf, 1e-300]:
        with pytest.raises(ValueError):
            LogNormalAFT(sigma)


def test_predictions_scale_monotonicity_and_fresh_process(tmp_path):
    import json
    import subprocess
    import sys

    p = fixture()
    fit = aft(p, p, context=RunContext("persist-aft", 1), sigma=0.7)
    model = AFTModel(fit.state.model, 0.7)
    times = [0.5, 2, 10, 100]
    probabilities = [0.1, 0.5, 0.9]
    raw = model.model.predict(p.data, offset=p.offset)[:, 0]
    got = model.predict(p.data, times=times, probabilities=probabilities, offset=p.offset)
    expected = aft_predict(raw, sigma=0.7, times=times, probabilities=probabilities)
    for key in got:
        np.testing.assert_allclose(got[key], expected[key], rtol=1e-12)
    assert np.all(np.diff(got["survival"], axis=1) <= 0)
    assert np.all(np.diff(got["quantile"], axis=1) > 0)
    path = tmp_path / "aft.json"
    model.save(path)
    assert AFTModel.load(path).identity == model.identity
    x = MixedData(
        [[1, "unknown"], [None, None]], [10, 11], p.data.feature_names, p.data.feature_kinds
    )
    code = """import json,sys
from openboost import MixedData
from openboost.survival import AFTModel
x=MixedData([[1,'unknown'],[None,None]],[10,11],('x','c'),('numeric','categorical'))
m=AFTModel.load(sys.argv[1])
print(json.dumps({k:v.tolist() for k,v in m.predict(x,times=[1,10],probabilities=[.2,.8],
offset=[[.1],[.2]]).items()}))
"""
    output = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    expected = model.predict(x, times=[1, 10], probabilities=[0.2, 0.8], offset=[[0.1], [0.2]])
    for key in expected:
        np.testing.assert_array_equal(output[key], expected[key])
    record = model.record()
    record["sigma"] = 0
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        AFTModel.load(path)


def test_rejected_update_and_all_censored_finite_initialization():
    p = fixture()
    from openboost.tree import depthwise

    def zero(data, fields):
        return depthwise(data, fields, max_depth=0, leaf=lambda *_: 0)

    result = aft(p, p, context=RunContext("reject", 1), learner=zero, step="backtracking")
    assert result.state.version == 0
    assert all(not s.accepted and s.raw_before is s.raw_after for s in result.steps)
    censored = replace(p, target=np.column_stack((p.target[:, 0], np.full(6, np.inf))))
    assert np.isfinite(LogNormalAFT().base(censored)).all()
