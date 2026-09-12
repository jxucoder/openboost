"""AFT scalar composition and scale-aware fresh CPU inference without device execution."""

import os
import subprocess
import sys

import numpy as np
import pytest

from openboost.artifacts import Model, TreeTerm
from openboost.binning import Binning
from openboost.stats import newton
from openboost.survival import AFTModel, LogNormalAFT
from openboost.tree import depthwise

from .reference.device_aft import ATOL, RTOL, rounds
from .reference.survival import aft_predict
from .test_device_aft_reference import fixture, original_rows


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def fresh_replay(artifact, validation, directory):
    path, inputs, output = (directory / name for name in ("aft.json", "aft-inputs.npz", "aft-output.npz"))
    artifact.save(path)
    restored = AFTModel.load(path)
    assert restored.identity == artifact.identity and restored.sigma == artifact.sigma
    np.savez(inputs, x=validation.data.values, offset=validation.offset)
    script = """
import sys
for name in ('cupy', 'numba', 'openboost.device_aft', 'openboost.device_runtime',
             'openboost.device_recipes', 'openboost.recipes', 'openboost.objectives'):
    sys.modules[name] = None
import numpy as np
from openboost import NumericData
from openboost.survival import AFTModel
model = AFTModel.load(sys.argv[1])
with np.load(sys.argv[2]) as f:
    data = NumericData(f['x'], np.arange(len(f['x'])), model.model.feature_names)
    prediction = model.predict(data, times=[.5, 2, 10, 100], probabilities=[.1, .5, .9], offset=f['offset'])
    np.savez(sys.argv[3], raw=model.model.predict(data, offset=f['offset']), **prediction)
"""
    subprocess.run([os.environ.get("OPENBOOST_FRESH_CPU_PYTHON", sys.executable), "-c", script,
                    str(path), str(inputs), str(output)], check=True, timeout=30)
    expected = artifact.predict(validation.data, times=[.5, 2, 10, 100], probabilities=[.1, .5, .9], offset=validation.offset)
    raw = artifact.model.predict(validation.data, offset=validation.offset)
    independent = aft_predict(raw[:, 0], sigma=artifact.sigma, times=[.5, 2, 10, 100], probabilities=[.1, .5, .9])
    with np.load(output) as replay:
        np.testing.assert_array_equal(replay["raw"], raw)
        for key, value in expected.items():
            np.testing.assert_array_equal(replay[key], value)
            close(value.reshape(independent[key].shape), independent[key])
    assert np.all(np.diff(expected["survival"], axis=1) <= 0)
    assert np.all(np.diff(expected["quantile"], axis=1) > 0)
    return restored


@pytest.mark.parametrize("sigma", [0.5, 0.7, 1, 2])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_two_prescribed_public_rounds_and_persisted_scale(sigma, depth, tmp_path):
    train, validation = fixture(), fixture(validation=True)
    initial, expected = rounds(original_rows(train), original_rows(validation), sigma=sigma, depth=depth)
    binning = Binning(("x",), (np.arange(7, dtype=float),))
    model = Model(("x",), [initial])
    objective = LogNormalAFT(sigma)
    for step in expected:
        raw = model.predict(train.data)
        close(raw[:, 0], step["before"])
        _, g, h = objective.geometry(train, raw)
        close(g, step["gradient"])
        close(h, step["curvature"])
        fields = newton(train, g, h)
        close(fields.values, step["fields"])
        tree = depthwise(binning.transform(train.data), fields, max_depth=depth)
        assert [None if f == -1 else (int(f), int(t), bool(m))
                for f, t, m in zip(tree.feature, tree.threshold, tree.missing_left, strict=True)] == [n["key"] for n in step["nodes"]]
        close(tree.value[:, 0], [n["value"] for n in step["nodes"]])
        model = Model(model.feature_names, model.base, (*model.terms, TreeTerm(tree, [[1]], 0.25)))
        close(model.predict(train.data)[:, 0], step["raw"])
        close(model.predict(validation.data)[:, 0], step["validation_raw"])
        close(objective.loss(train, model.predict(train.data)), step["loss"])
        close(objective.loss(validation, model.predict(validation.data)), step["score"])
    artifact = AFTModel(model, sigma)
    fresh_replay(artifact, validation, tmp_path)
    wrong = AFTModel(model, 3 * sigma)
    times = [.5, 2, 10, 100]
    assert not np.allclose(artifact.predict(validation.data, times=times)["survival"], wrong.predict(validation.data, times=times)["survival"])


def test_aft_export_rejects_non_run_before_device_work():
    from openboost import device_aft
    with pytest.raises(ValueError, match="DeviceRun"):
        device_aft.export(None, None)
