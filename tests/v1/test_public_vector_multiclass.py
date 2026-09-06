"""Joint vector topology and multiclass conformance using independent oracles."""

import json
import subprocess
import sys

import numpy as np
import pytest

from openboost import ClassSchema, MixedData, Problem, RunContext
from openboost.artifacts import Model, TreeTerm
from openboost.binning import Binning
from openboost.objectives import Multiclass
from openboost.ops import vector_feasible, vector_leaf, vector_score
from openboost.recipes import multiclass
from openboost.stats import vector_newton
from openboost.tree import Tree, best_first, depthwise, symmetric
from tests.v1.reference.classification import softmax
from tests.v1.reference.mixed import Transformer, grow


def fixture():
    x = MixedData(
        [
            [0, "a"],
            [1, "b"],
            [2, "c"],
            [3, None],
            [4, "a"],
            [None, "b"],
            [1, "c"],
            [2, "a"],
            [3, "b"],
        ],
        np.arange(9),
        ("x", "c"),
        ("numeric", "categorical"),
    )
    schema = ClassSchema.fit(["a", "b", "c"])
    p = Problem(
        x,
        schema.encode(["a", "b", "c", "c", "a", "b", "c", "a", "b"]),
        x.row_ids,
        weight=[1, 0, 2, 1, 3, 1, 2, 1, 1],
        raw_width=3,
        classes=schema,
    )
    ref = Transformer.fit(x.values, names=x.feature_names, kinds=x.feature_kinds, bins=4)
    return p, Binning.fit(x, bins=4).transform(x), ref


@pytest.mark.parametrize("policy", [depthwise, best_first, symmetric])
@pytest.mark.parametrize("projected", [False, True])
def test_vector_tree_and_full_leaves_match_independent_oracle(policy, projected):
    p, b, ref_transform = fixture()
    rng = np.random.default_rng(27)
    g, h = rng.normal(size=(9, 3)), rng.uniform(0.5, 2, (9, 3))
    projection = np.array([[1.0], [0], [0]]) if projected else np.eye(3)
    fields = vector_newton(p, g @ projection, h @ (projection**2))
    leaves = vector_newton(p, g, h)
    tree = policy(
        b,
        fields,
        max_depth=3,
        scoring=vector_score,
        legality=vector_feasible,
        leaf=vector_leaf,
        leaf_fields=leaves,
    )
    if projected:
        full = policy(
            b, leaves, max_depth=3, scoring=vector_score, legality=vector_feasible, leaf=vector_leaf
        )
        assert tree.identity != full.identity
    expected = grow(
        p.data.values,
        g,
        h,
        ref_transform,
        weight=p.weight,
        projection=projection,
        policy=policy.__name__,
        max_depth=3,
    )
    assert tree.output_width == 3 and len(tree.value) == len(expected.nodes)
    for i, node in enumerate(expected.nodes):
        np.testing.assert_allclose(tree.value[i], node.value)
        assert (tree.left[i], tree.right[i]) == (node.left, node.right)
        if node.condition:
            assert (tree.feature[i], tree.threshold[i], tree.missing_left[i]) == node.condition
    np.testing.assert_allclose(tree.predict(p.data), expected.predict(p.data.values))
    # Learner width L can differ from model raw width K.
    term = TreeTerm(tree, [[1, 0], [0, 1], [1, -1]], 0.5)
    model = Model(p.data.feature_names, [0, 1], (term,))
    np.testing.assert_allclose(
        model.predict(p.data), [0, 1] + 0.5 * tree.predict(p.data) @ term.mapping
    )


def test_three_round_multiclass_geometry_and_joint_tree_oracle():
    p, b, transform = fixture()
    fit = multiclass(p, p, context=RunContext("softmax", 1), rounds=3, bins=4)
    raw = np.zeros((9, 3))
    for actual in fit.steps:
        loss, _prob, g, exact, bound = softmax(raw, p.target[:, 0], weight=p.weight)
        np.testing.assert_allclose(actual.gradient, g)
        np.testing.assert_allclose(actual.diagonal_bound, bound)
        assert np.linalg.eigvalsh(np.array([np.diag(row) for row in bound]) - exact).min() > -1e-12
        tree = grow(p.data.values, g, bound, transform, weight=p.weight)
        np.testing.assert_allclose(actual.raw_before, raw)
        raw += 0.1 * tree.predict(p.data.values)
        np.testing.assert_allclose(actual.raw_after, raw)
        np.testing.assert_allclose(
            [actual.loss_before, actual.loss_after],
            [loss, softmax(raw, p.target[:, 0], weight=p.weight)[0]],
        )
    assert fit.state.version == 3 and len(fit.state.model.terms) == 3
    assert all(t.learner.output_width == 3 for t in fit.state.model.terms)
    np.testing.assert_allclose(
        fit.state.model.predict_proba(p.data), softmax(raw, p.target[:, 0])[1]
    )


def test_vector_classifier_fresh_process_and_corrupt_payload(tmp_path):
    p, _b, _ref = fixture()
    model = multiclass(p, p, context=RunContext("persist", 1), rounds=2).state.model
    path = tmp_path / "model.json"
    model.save(path)
    assert Model.load(path).identity == model.identity
    code = """import json, sys
from openboost import MixedData
from openboost.artifacts import Model
x = MixedData([[-10,'a'],[100,'unknown'],[None,None]], [10,11,12], ('x','c'), ('numeric','categorical'))
m = Model.load(sys.argv[1])
print(json.dumps([m.predict_proba(x).tolist(), m.predict_label(x)]))
"""
    output = json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)], text=True))
    x = MixedData(
        [[-10, "a"], [100, "unknown"], [None, None]],
        [10, 11, 12],
        ("x", "c"),
        ("numeric", "categorical"),
    )
    np.testing.assert_array_equal(output[0], model.predict_proba(x))
    assert tuple(output[1]) == model.predict_label(x)
    for corrupt in ("width", "nan", "mapping"):
        record = model.record()
        if corrupt == "mapping":
            record["terms"][0]["mapping"] = [[1, 0, 0]]
        else:
            record["terms"][0]["learner"]["value"][-1] = (
                [1] if corrupt == "width" else [float("nan")] * 3
            )
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            Model.load(path)


def test_offsets_joint_rejection_and_foreign_leaf_fields():
    p, b, _ref = fixture()
    from dataclasses import replace

    shifted = replace(p, offset=np.arange(27).reshape(9, 3) / 10)
    raw = np.zeros((9, 3))
    got = Multiclass.geometry(shifted, raw)
    ref = softmax(shifted.with_offset(raw), p.target[:, 0], weight=p.weight)
    for a, expected in zip(got, (ref[0], ref[2], ref[4]), strict=True):
        np.testing.assert_allclose(a, expected)

    def zero(data, fields):
        return depthwise(data, fields, max_depth=0, leaf=lambda *_: [0, 0, 0])

    fit = multiclass(
        p, p, context=RunContext("reject", 1), learner=zero, rounds=2, step="backtracking"
    )
    assert fit.state.version == 0 and not fit.state.model.terms
    assert all(not item.accepted and item.raw_before is item.raw_after for item in fit.steps)
    fields = vector_newton(p, np.ones((9, 3)), np.ones((9, 3)))
    other = vector_newton(shifted, np.ones((9, 3)), np.ones((9, 3)))
    with pytest.raises(ValueError, match="same problem"):
        depthwise(b, fields, leaf_fields=other, leaf=vector_leaf)
    with pytest.raises(ValueError):
        Tree.from_record({"format": "openboost-tree-v2"})
