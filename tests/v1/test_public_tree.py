"""Depthwise trees checked against independent exhaustive original-row growth."""

import json
import os
import subprocess
import sys
from functools import partial

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost.binning import Binning
from openboost.ops import feasible, newton_leaf, score
from openboost.stats import newton
from openboost.tree import Tree, depthwise
from tests.v1.reference.tree import fit_tree


def fixture():
    x = NumericData(
        [[0, 2], [1, 0], [2, 1], [3, 3], [4, np.nan], [np.nan, 1], [6, 2], [7, 0]],
        np.arange(8) + 40,
        ("a", "b"),
    )
    p = Problem(x, np.zeros((8, 1)), x.row_ids, weight=[1, 0, 2, 1, 3, 1, 1, 2])
    g, h = np.array([-8, -3, -2, -1, 1, 3, 5, 8.0]), np.ones(8)
    b = Binning.fit(x, bins=8).transform(x)
    return p, b, g, h


@pytest.mark.parametrize("depth,leaves", [(0, 1), (1, 8), (2, 3), (3, 5), (8, None)])
def test_topology_and_inference_match_exhaustive_reference(depth, leaves):
    p, b, g, h = fixture()
    actual = depthwise(b, newton(p, g, h), max_depth=depth, max_leaves=leaves)
    ref = fit_tree(
        np.where(b.missing.T, np.nan, b.codes.T),
        g,
        h,
        weight=p.weight,
        max_depth=depth,
        max_leaves=leaves,
    )
    assert len(actual.value) == len(ref.nodes)
    for i, node in enumerate(ref.nodes):
        assert actual.value[i] == pytest.approx(node.value)
        assert (actual.left[i], actual.right[i]) == (node.left, node.right)
        if node.condition is None:
            assert actual.feature[i] == -1
        else:
            assert (actual.feature[i], actual.threshold[i], actual.missing_left[i]) == (
                node.condition.feature,
                node.condition.threshold,
                node.condition.missing_left,
            )
    unseen = NumericData(
        [[-100, 0], [100, 2], [np.nan, np.nan], [3.5, 1]], [1, 2, 3, 4], ("a", "b")
    )
    encoded = b.binning.transform(unseen)
    np.testing.assert_allclose(
        actual.predict(unseen)[:, 0],
        ref.predict(np.where(encoded.missing.T, np.nan, encoded.codes.T)),
    )
    with pytest.raises(ValueError):
        actual.left.flags.writeable = True


def test_custom_callbacks_change_tree_and_leaf_values():
    p, b, g, h = fixture()
    fields = newton(p, g, h).add_independent("mass", np.ones(8))
    calls = {"score": 0, "legal": 0, "leaf": 0}

    def scoring(c):
        calls["score"] += 1
        return score(c) if c.feature == 1 else 0

    def legality(c):
        calls["legal"] += 1
        return feasible(c, min_information={"mass": 2})

    def leaf(total, names):
        calls["leaf"] += 1
        return 2 * newton_leaf(total, names)

    custom = depthwise(b, fields, scoring=scoring, legality=legality, leaf=leaf)
    assert all(calls.values()) and calls["leaf"] == len(custom.value)
    assert set(custom.feature) == {-1, 1}
    assert custom.identity != depthwise(b, fields).identity
    expected = fit_tree(
        np.where(b.missing.T, np.nan, b.codes.T),
        g,
        h,
        weight=p.weight,
        information=np.ones((8, 1)),
        min_information=2,
    )
    constrained = depthwise(b, fields, legality=partial(feasible, min_information={"mass": 2}))
    np.testing.assert_allclose(
        constrained.predict(p.data)[:, 0],
        expected.predict(np.where(b.missing.T, np.nan, b.codes.T)),
    )
    baseline = depthwise(b, fields, scoring=scoring, legality=legality)
    np.testing.assert_allclose(custom.predict(p.data), 2 * baseline.predict(p.data))


def test_tree_roundtrip_in_fresh_process(tmp_path):
    p, b, g, h = fixture()
    tree = depthwise(b, newton(p, g, h))
    path = tmp_path / "tree.json"
    tree.save(path)
    restored = Tree.load(path)
    assert restored.identity == tree.identity
    data = NumericData([[-100, 0], [100, 2], [np.nan, np.nan]], [1, 2, 3], ("a", "b"))
    np.testing.assert_array_equal(restored.predict(data), tree.predict(data))
    code = """import json, sys
import numpy as np
from openboost import NumericData
from openboost.tree import Tree
x = NumericData([[-100, 0], [100, 2], [np.nan, np.nan]], [1, 2, 3], ("a", "b"))
print(json.dumps(Tree.load(sys.argv[1]).predict(x).tolist()))
"""
    output = subprocess.check_output(
        [sys.executable, "-c", code, str(path)], env=os.environ, text=True
    )
    np.testing.assert_array_equal(json.loads(output), tree.predict(data))
    wrong = NumericData(data.values, data.row_ids, ("b", "a"))
    with pytest.raises(ValueError):
        tree.predict(wrong)


@pytest.mark.parametrize(
    "corruption",
    [
        "cycle",
        "shared",
        "outside",
        "fraction",
        "schema",
        "nan",
        "leaf",
        "cuts",
        "route",
        "unreachable",
        "unknown",
    ],
)
def test_corrupt_artifacts_rejected(tmp_path, corruption):
    p, b, g, h = fixture()
    record = depthwise(b, newton(p, g, h)).record()
    if corruption == "cycle":
        record["left"][0] = 0
    elif corruption == "shared":
        record["right"][0] = record["left"][0]
    elif corruption == "outside":
        record["left"][0] = 2**31
    elif corruption == "fraction":
        record["left"][0] = 1.5
    elif corruption == "schema":
        record["feature"][0] = 2
    elif corruption == "nan":
        record["value"][0] = float("nan")
    elif corruption == "leaf":
        record["left"][-1] = 0
    elif corruption == "cuts":
        record["cuts"][0] = [2, 1]
    elif corruption == "route":
        record["missing_left"][0] = 1
    elif corruption == "unreachable":
        for name, value in zip(
            ("feature", "threshold", "missing_left", "left", "right", "value"),
            (-1, -1, False, -1, -1, 0),
            strict=True,
        ):
            record[name].append(value)
    else:
        record["extra"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        Tree.load(path)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_depth": -1},
        {"max_depth": True},
        {"max_leaves": 0},
        {"max_leaves": 2**31},
        {"leaf": lambda *_: np.nan},
        {"scoring": lambda _: np.nan},
    ],
)
def test_bad_configuration_and_callbacks_rejected(kwargs):
    p, b, g, h = fixture()
    with pytest.raises(ValueError):
        depthwise(b, newton(p, g, h), **kwargs)


def test_score_once_and_empty_child_guard():
    p, b, g, h = fixture()
    seen = set()

    def scoring(c):
        key = (c.rows_identity, c.key)
        assert key not in seen
        seen.add(key)
        return score(c)

    depthwise(b, newton(p, g, h), scoring=scoring, max_leaves=3)
    assert seen
    constant = NumericData([[1], [1]], [1, 2], ("x",))
    problem = Problem(constant, [[0], [0]], constant.row_ids)
    with pytest.raises(ValueError, match="empty child"):
        depthwise(
            Binning.fit(constant).transform(constant),
            newton(problem, [-1, 1], [1, 1]),
            legality=lambda _: True,
            scoring=lambda _: 1,
        )


def test_duplicate_fields_and_root_only_artifact(tmp_path):
    p, b, g, h = fixture()
    tree = depthwise(b, newton(p, g, h), max_depth=0)
    path = tmp_path / "tree.json"
    tree.save(path)
    assert Tree.load(path).identity == tree.identity
    path.write_text(path.read_text().replace('"format":', '"format": "duplicate", "format":'))
    with pytest.raises(ValueError, match="duplicate"):
        Tree.load(path)
