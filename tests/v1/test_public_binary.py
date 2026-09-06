"""Typed binary labels, stable geometry and complete classification inference."""

from openboost import ClassSchema


def test_training_class_schema():
    schema = ClassSchema.fit(["yes", "no", "yes"])
    assert schema.values == ("no", "yes")
    assert schema.encode(["yes", "no"]).tolist() == [[1], [0]]


import json
import subprocess
import sys

import numpy as np
import pytest

from openboost import MixedData, NumericData, Problem, RunContext
from openboost.artifacts import Model
from openboost.binning import Binning
from openboost.objectives import Binary, Squared
from openboost.recipes import binary
from openboost.runtime import initialize
from tests.v1.reference.classification import ClassMap, binary_base
from tests.v1.reference.classification import binary as ref_binary
from tests.v1.reference.tree import fit_tree


def fixture():
    x = NumericData([[0], [1], [2], [3], [4], [np.nan]], np.arange(6), ("x",))
    labels = ["no", "yes", "no", "yes", "yes", "no"]
    schema = ClassSchema.fit(labels)
    return Problem(x, schema.encode(labels), x.row_ids, weight=[1, 0, 2, 1, 3, 1], classes=schema)


def test_three_rounds_match_independent_logistic_reference():
    p = fixture()
    result = binary(p, p, context=RunContext("logistic", 7), rounds=3, bins=6)
    base = binary_base(p.target[:, 0], weight=p.weight, clip=1e-6)
    assert result.state.model.base[0] == pytest.approx(base)
    raw = np.full(6, base)
    b = Binning.fit(p.data, bins=6).transform(p.data)
    bins = np.where(b.missing.T, np.nan, b.codes.T)
    for actual in result.steps:
        loss, g, h = ref_binary(raw, p.target[:, 0], weight=p.weight)
        np.testing.assert_allclose(actual.gradient, g)
        np.testing.assert_allclose(actual.curvature, h)
        np.testing.assert_allclose(actual.raw_before[:, 0], raw)
        tree = fit_tree(bins, g, h, weight=p.weight)
        raw = raw + 0.1 * tree.predict(bins)
        np.testing.assert_allclose(actual.raw_after[:, 0], raw)
        np.testing.assert_allclose(
            [actual.loss_before, actual.loss_after],
            [loss, ref_binary(raw, p.target[:, 0], weight=p.weight)[0]],
        )
    assert result.state.model.classes == p.classes
    assert result.steps[-1].loss_after < result.steps[0].loss_before


def test_extreme_logits_and_offset_geometry():
    p = fixture()
    p = Problem(
        p.data,
        p.target,
        p.row_ids,
        weight=p.weight,
        classes=p.classes,
        offset=np.arange(6)[:, None] / 10,
    )
    raw = np.array([[-1000], [1000], [-40], [40], [0], [1.0]])
    got = Binary.geometry(p, raw)
    ref = ref_binary(p.with_offset(raw)[:, 0], p.target[:, 0], weight=p.weight)
    for a, b in zip(got, ref, strict=True):
        np.testing.assert_allclose(a, b)
    assert got[1][3] < 0 and got[2][3] > 0  # Tiny tails must not cancel to zero.
    model = binary(p, p, context=RunContext("offset", 1), rounds=2).state.model
    probability = model.predict_proba(p.data, offset=p.offset)
    np.testing.assert_allclose(probability.sum(axis=1), 1)
    from openboost.outputs import binary_probabilities

    np.testing.assert_array_equal(
        probability, binary_probabilities(model.predict(p.data) + p.offset)
    )


def test_class_schema_matches_reference_and_rejects_wrong_roles():
    p = fixture()
    ref = ClassMap.fit(["yes", "no", "yes"])
    assert p.classes.values == ref.values
    np.testing.assert_array_equal(p.classes.encode(["yes", "no"])[:, 0], ref.encode(["yes", "no"]))
    assert p.classes.decode([1, 0]) == ref.decode([1, 0])
    for labels in (["unknown"], [None], [True]):
        with pytest.raises(ValueError):
            p.classes.encode(labels)
    for labels in (["only"], [1, "two"], [None, 1]):
        with pytest.raises(ValueError):
            ClassSchema.fit(labels)
    with pytest.raises(ValueError):
        Problem(p.data, [[2]] * 6, p.row_ids, classes=p.classes)
    with pytest.raises(ValueError):
        Squared.validate(p)
    other = Problem(p.data, p.target, p.row_ids, classes=ClassSchema(("a", "b")))
    with pytest.raises(ValueError, match="schemas"):
        binary(p, other, context=RunContext("mismatch", 1), rounds=0)
    assert other.identity != p.identity


def test_fresh_process_classified_mixed_model_and_corrupt_schema(tmp_path):
    x = MixedData(
        [[0, "a"], [1, "b"], [2, "a"], [3, None]],
        [1, 2, 3, 4],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    schema = ClassSchema.fit([10, 20])
    p = Problem(x, schema.encode([10, 20, 10, 20]), x.row_ids, classes=schema)
    model = binary(p, p, context=RunContext("persist", 1), rounds=3).state.model
    path = tmp_path / "classifier.json"
    model.save(path)
    assert Model.load(path).identity == model.identity
    code = """import json, sys
from openboost import MixedData
from openboost.artifacts import Model
x = MixedData([[0,'a'],[100,'unknown'],[None,None]], [10,11,12], ('x','c'), ('numeric','categorical'))
m = Model.load(sys.argv[1])
print(json.dumps([m.predict_proba(x).tolist(), m.predict_label(x)]))
"""
    output = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    unseen = MixedData(
        [[0, "a"], [100, "unknown"], [None, None]], [10, 11, 12], x.feature_names, x.feature_kinds
    )
    np.testing.assert_array_equal(output[0], model.predict_proba(unseen))
    assert tuple(output[1]) == model.predict_label(unseen)
    for classes in ([20, 10], [10, 10], [10, "20"], [10, 20, 30]):
        record = model.record()
        record["classes"] = classes
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            Model.load(path)


def test_binary_rejection_best_state_and_initialization_contract():
    p = fixture()
    from openboost.tree import depthwise

    def zero(data, fields):
        return depthwise(data, fields, max_depth=0, leaf=lambda *_: 0)

    result = binary(
        p, p, context=RunContext("reject", 1), rounds=2, learner=zero, step="backtracking"
    )
    assert result.state.version == 0 and result.state.best_model is result.state.model
    assert all(not item.accepted and item.raw_before is item.raw_after for item in result.steps)
    valid = Problem(p.data, 1 - p.target, p.row_ids, weight=p.weight, classes=p.classes)
    result = binary(p, valid, context=RunContext("best", 1), rounds=3)
    assert result.state.best_model is not result.state.model
    assert result.state.best_model.classes == p.classes
    one = Problem(p.data, np.zeros((6, 1)), p.row_ids, classes=p.classes)
    with pytest.raises(ValueError, match="both classes"):
        binary(one, p, context=RunContext("one", 1), rounds=0)
    with pytest.raises(ValueError):
        binary(p, p, context=RunContext("clip", 1), rounds=0, clip=1e-100)
    initial = initialize(RunContext("ties", 1), p, p, [0], score=Binary.loss)
    assert set(initial.model.predict_label(p.data)) == {p.classes.values[0]}
