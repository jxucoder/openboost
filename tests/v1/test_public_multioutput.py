"""Multi-output regression policies, scaling and persisted original units."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import MixedData, Problem, RunContext
from openboost.multioutput import MultiOutputModel, TargetScale
from openboost.objectives import MultiSquared
from openboost.recipes import multi_squared, squared
from openboost.tree import best_first, depthwise, symmetric
from tests.v1.reference.mixed import Transformer, grow


def fixture():
    x = MixedData(
        [[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "c"], [None, "b"]],
        [0, 1, 2, 3, 4, 5],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    return Problem(
        x,
        [[-5, 3], [2, 8], [8, -4], [3, 2], [-2, 9], [15, -1]],
        x.row_ids,
        weight=[1, 3, 2, 0, 4, 2],
        offset=np.arange(12.0).reshape(6, 2) / 10,
    )


@pytest.mark.parametrize("policy", [depthwise, best_first, symmetric])
@pytest.mark.parametrize("mode", ["shared", "projected", "independent"])
def test_three_rounds_match_independent_reference(policy, mode):
    p = fixture()
    projection = np.array([[1.0], [0.0]]) if mode == "projected" else None
    actual = multi_squared(
        p,
        p,
        context=RunContext("multi", 1),
        rounds=3,
        bins=4,
        grower=policy,
        mode="independent" if mode == "independent" else "shared",
        projection=projection,
    )
    raw = np.broadcast_to(np.average(p.target - p.offset, axis=0, weights=p.weight), (6, 2)).copy()
    transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    for step in actual.steps:
        g = p.with_offset(raw) - p.target
        np.testing.assert_allclose(step.gradient, g)
        if mode == "independent":
            delta = np.column_stack(
                [
                    grow(
                        p.data.values,
                        g[:, k : k + 1],
                        np.ones((6, 1)),
                        transform,
                        weight=p.weight,
                        policy=policy.__name__,
                    ).predict(p.data.values)[:, 0]
                    for k in range(2)
                ]
            )
        else:
            delta = grow(
                p.data.values,
                g,
                np.ones_like(g),
                transform,
                weight=p.weight,
                policy=policy.__name__,
                projection=projection,
            ).predict(p.data.values)
        np.testing.assert_allclose(step.raw_before, raw)
        raw += 0.1 * delta
        np.testing.assert_allclose(step.raw_after, raw)
        expected = np.average((p.with_offset(raw) - p.target) ** 2, axis=0, weights=p.weight)
        np.testing.assert_allclose(step.mse_after, expected)
    assert actual.state.version == 3
    assert len(actual.state.model.terms) == (6 if mode == "independent" else 3)


def test_output_permutation_and_single_output_limit():
    p = fixture()
    left = multi_squared(p, p, context=RunContext("left", 1), rounds=2)
    reversed_p = replace(p, target=p.target[:, ::-1], offset=p.offset[:, ::-1])
    right = multi_squared(reversed_p, reversed_p, context=RunContext("right", 1), rounds=2)
    np.testing.assert_allclose(left.state.train_raw[:, ::-1], right.state.train_raw)
    single = replace(p, target=p.target[:, :1], offset=p.offset[:, :1], raw_width=1)
    regular = squared(single, single, context=RunContext("scalar", 1))
    for mode in ("shared", "independent"):
        result = multi_squared(single, single, context=RunContext(mode, 1), mode=mode)
        np.testing.assert_allclose(result.state.train_raw, regular.state.train_raw)


def test_training_scaling_constant_and_original_units():
    p = fixture()
    scale = TargetScale.fit(p)
    np.testing.assert_allclose(scale.mean, np.average(p.target, weights=p.weight, axis=0))
    transformed = scale.transform(p)
    np.testing.assert_allclose(
        np.average(transformed.target, weights=p.weight, axis=0), 0, atol=1e-15
    )
    valid = replace(p, target=p.target + 100)
    np.testing.assert_allclose(
        scale.transform(valid).target, (valid.target - scale.mean) / scale.scale
    )
    constant = replace(p, target=np.column_stack((p.target[:, 0], np.full(6, 7.0))))
    constant_scale = TargetScale.fit(constant)
    assert constant_scale.constant == (False, True) and constant_scale.scale[1] == 1
    fit = multi_squared(transformed, transformed, context=RunContext("scaled", 1))
    model = MultiOutputModel(fit.state.model, scale)
    np.testing.assert_allclose(
        model.predict(p.data, offset=p.offset),
        fit.state.model.predict(p.data) * scale.scale + scale.mean + p.offset,
    )


def test_wrong_target_kind_projection_and_atomic_rejection():
    p = fixture()
    censored = replace(p, target=np.ones((6, 2)), target_kind="event_right")
    with pytest.raises(ValueError):
        MultiSquared.validate(censored)
    for projection, mode in [
        (np.zeros((2, 1)), "shared"),
        (np.ones((3, 1)), "shared"),
        (np.ones((2, 1)), "independent"),
    ]:
        with pytest.raises(ValueError):
            multi_squared(p, p, context=RunContext("bad", 1), projection=projection, mode=mode)
    result = multi_squared(
        p, p, context=RunContext("reject", 1), learning_rate=0, step="backtracking", rounds=2
    )
    assert result.state.version == 0
    assert all(not s.accepted and s.raw_before is s.raw_after for s in result.steps)


@pytest.mark.parametrize("mode", ["shared", "independent"])
def test_fresh_process_scaled_roundtrip(tmp_path, mode):
    import json
    import subprocess
    import sys

    p = fixture()
    scale = TargetScale.fit(p)
    q = scale.transform(p)
    fit = multi_squared(q, q, context=RunContext("persist", 1), mode=mode)
    model = MultiOutputModel(fit.state.model, scale)
    path = tmp_path / "multi.json"
    model.save(path)
    assert MultiOutputModel.load(path).identity == model.identity
    x = MixedData(
        [[1, "unknown"], [None, None]], [10, 11], p.data.feature_names, p.data.feature_kinds
    )
    code = """import json,sys
from openboost import MixedData
from openboost.multioutput import MultiOutputModel
x=MixedData([[1,'unknown'],[None,None]],[10,11],('x','c'),('numeric','categorical'))
print(json.dumps(MultiOutputModel.load(sys.argv[1]).predict(x,offset=[[.1,.2],[.3,.4]]).tolist()))
"""
    output = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    np.testing.assert_array_equal(output, model.predict(x, offset=[[0.1, 0.2], [0.3, 0.4]]))
    record = model.record()
    record["scale"] = [0, 1]
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        MultiOutputModel.load(path)
