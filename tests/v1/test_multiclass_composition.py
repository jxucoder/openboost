"""Independent separate-topology trajectories against actual public CPU operations."""

import os
import subprocess
import sys

import numpy as np
import pytest

from openboost.artifacts import Model, TreeTerm
from openboost.binning import Binning
from openboost.objectives import Multiclass
from openboost.stats import newton
from openboost.tree import depthwise

from .reference.device_multiclass import ATOL, RTOL, geometry, rounds
from .reference.device_splits import enumerate_candidates
from .test_device_multiclass_reference import fixture


def original_rows(problem):
    return dict(
        x=problem.data.values,
        target=problem.target[:, 0],
        offset=problem.offset,
        weight=problem.weight,
    )


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL)


def topology(tree):
    return [
        None if f == -1 else (int(f), int(t), bool(m))
        for f, t, m in zip(tree.feature, tree.threshold, tree.missing_left, strict=True)
    ]


def fresh_replay(model, problem, directory):
    """CPU-only process with no objective, device runtime or training dependency."""
    path, inputs, output = (directory / name for name in ("model.json", "inputs.npz", "output.npz"))
    model.save(path)
    restored = Model.load(path)
    assert restored.identity == model.identity and restored.classes == problem.classes
    np.savez(inputs, x=problem.data.values, offset=problem.offset)
    script = """
import sys
for name in ('cupy', 'numba', 'openboost.objectives', 'openboost.device_multiclass',
             'openboost.device_runtime', 'openboost.recipes'):
    sys.modules[name] = None
import numpy as np
from openboost import NumericData
from openboost.artifacts import Model
model = Model.load(sys.argv[1])
with np.load(sys.argv[2]) as f:
    data = NumericData(f['x'], np.arange(len(f['x'])), model.feature_names)
    np.savez(sys.argv[3], raw=model.predict(data, offset=f['offset']),
             probability=model.predict_proba(data, offset=f['offset']),
             labels=model.predict_label(data, offset=f['offset']))
"""
    subprocess.run(
        [os.environ.get("OPENBOOST_FRESH_CPU_PYTHON", sys.executable),
         "-c", script, str(path), str(inputs), str(output)], check=True, timeout=30
    )
    with np.load(output) as replay:
        for name, expected in (
            ("raw", model.predict(problem.data, offset=problem.offset)),
            ("probability", model.predict_proba(problem.data, offset=problem.offset)),
            ("labels", model.predict_label(problem.data, offset=problem.offset)),
        ):
            np.testing.assert_array_equal(replay[name], expected)
    return restored


@pytest.mark.parametrize("width", [2, 3, 5])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_two_round_public_cpu_composition_and_persisted_class_order(width, depth, tmp_path):
    train, validation = fixture(width), fixture(width, validation=True)
    initial, expected = rounds(original_rows(train), original_rows(validation), depth=depth)
    binning = Binning(("x",), (np.arange(len(train.target) - 1, dtype=float),))
    binned = binning.transform(train.data)
    model = Model(("x",), initial, classes=train.classes)
    for step in expected:
        raw = model.predict(train.data)
        close(raw, step["before"])
        loss, g, h = Multiclass.geometry(train, raw)
        close(g, step["gradient"])
        close(h, step["bound"])
        terms = []
        for j, nodes in enumerate(step["trees"]):
            fields = newton(train, g[:, j], h[:, j])
            close(fields.values, step["fields"][j])
            if depth:
                candidates, _ = enumerate_candidates(
                    train.data.values, step["fields"][j], range(len(raw))
                )
                gains = sorted((c["gain"] for c in candidates if c["legal"]), reverse=True)
                assert gains[0] - gains[1] > 1e-3
            tree = depthwise(binned, fields, max_depth=depth)
            assert topology(tree) == [n["key"] for n in nodes]
            close(tree.value[:, 0], [n["value"] for n in nodes])
            terms.append(TreeTerm(tree, np.eye(width)[j : j + 1], 0.25))
        model = Model(
            model.feature_names, model.base, (*model.terms, *terms), classes=model.classes
        )
        close(model.predict(train.data), step["raw"])
        close(model.predict(validation.data), step["validation_raw"])
        assert Multiclass.loss(train, model.predict(train.data)) < loss
        close(Multiclass.loss(validation, model.predict(validation.data)), step["score"])
    assert len(model.terms) == 2 * width
    assert not np.array_equal(expected[0]["gradient"], expected[1]["gradient"])
    probabilities = geometry(
        expected[-1]["validation_raw"],
        validation.target[:, 0],
        validation.offset,
        validation.weight,
    )[3]
    close(model.predict_proba(validation.data, offset=validation.offset), probabilities)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-15)
    np.testing.assert_array_equal(
        model.predict_label(validation.data, offset=validation.offset),
        train.classes.decode(probabilities.argmax(axis=1)),
    )
    fresh_replay(model, validation, tmp_path)
