"""Mixed input, category equality, missing routes and inference conformance."""

from openboost import MixedData


def test_owned_mixed_input():
    values = [[0, "b"], [1, "a"], [2, None]]
    data = MixedData(values, [1, 2, 3], ("x", "c"), ("numeric", "categorical"))
    values[0][1] = "changed"
    assert data.values[0, 1] == "b"


import json
import subprocess
import sys

import numpy as np
import pytest

from openboost import NumericData, Problem, RunContext
from openboost.artifacts import Model
from openboost.binning import Binning
from openboost.ops import candidates, choose, histogram, partition
from openboost.recipes import squared
from openboost.stats import newton
from openboost.tree import Tree, best_first, depthwise, symmetric
from tests.v1.reference.data import CategoryMap
from tests.v1.reference.mixed import Transformer, grow


def fixture():
    values = [[0, "c"], [1, "a"], [2, "b"], [3, None], [4, "a"], [5, "c"], [np.nan, "b"], [1, "a"]]
    data = MixedData(values, np.arange(8), ("x", "c"), ("numeric", "categorical"))
    p = Problem(data, np.zeros((8, 1)), data.row_ids, weight=[1, 0, 2, 1, 3, 1, 2, 1])
    return p, np.array([-4, -1, 4, -2, -3, -4, 4, -1.0])


@pytest.mark.parametrize("policy", [depthwise, best_first, symmetric])
def test_mixed_topology_matches_raw_reference(policy):
    p, gradient = fixture()
    fitted = Binning.fit(p.data, bins=4)
    b = fitted.transform(p.data)
    tree = policy(b, newton(p, gradient, np.ones(8)), max_depth=3)
    ref_transform = Transformer.fit(
        p.data.values, names=p.data.feature_names, kinds=p.data.feature_kinds, bins=4
    )
    ref = grow(
        p.data.values,
        gradient[:, None],
        np.ones((8, 1)),
        ref_transform,
        weight=p.weight,
        policy=policy.__name__,
        max_depth=3,
    )
    assert len(tree.value) == len(ref.nodes)
    assert any(tree.feature == 1)
    for i, node in enumerate(ref.nodes):
        assert tree.value[i] == pytest.approx(node.value[0])
        assert (tree.left[i], tree.right[i]) == (node.left, node.right)
        if node.condition:
            assert (tree.feature[i], tree.threshold[i], tree.missing_left[i]) == node.condition
    unseen = MixedData(
        [[-100, "a"], [100, "b"], [None, "unseen"], [2, None]],
        [90, 91, 92, 93],
        p.data.feature_names,
        p.data.feature_kinds,
    )
    np.testing.assert_allclose(tree.predict(unseen), ref.predict(unseen.values))


@pytest.mark.parametrize("tokens", [["b", "a", None, "c"], [3, 1, None, 2], [None, None]])
def test_dictionary_matches_independent_map_and_unknown_is_missing(tokens):
    data = MixedData([[v] for v in tokens], np.arange(len(tokens)), ("c",), ("categorical",))
    fitted = Binning.fit(data)
    ref = CategoryMap.fit(tokens)
    assert fitted.categories[0] == ref.values
    unseen = MixedData(
        [[tokens[0]], [None], ["unknown"]]
        if isinstance(tokens[0], str) or tokens[0] is None
        else [[tokens[0]], [None], [99]],
        [1, 2, 3],
        ("c",),
        ("categorical",),
    )
    b = fitted.transform(unseen)
    code, missing = ref.transform(unseen.values[:, 0])
    np.testing.assert_array_equal(b.codes[0], code)
    np.testing.assert_array_equal(b.missing[0], missing)


@pytest.mark.parametrize("missing_left", [False, True])
def test_category_equality_routes_both_missing_directions(missing_left):
    p, gradient = fixture()
    b = Binning.fit(p.data, bins=4).transform(p.data)
    option = next(
        c
        for c in candidates(histogram(b, newton(p, gradient, np.ones(8))))
        if c.feature == 1 and c.threshold == 1 and c.missing_left == missing_left
    )
    assert option.kind == "categorical"
    left, right = partition(b, None, option)
    expected = [2, 3, 6] if missing_left else [2, 6]
    assert list(left) == expected
    assert sorted(list(left) + list(right)) == list(range(8))
    np.testing.assert_allclose(option.left[0], np.dot(p.weight[left], gradient[left]))


def test_categorical_recipe_roundtrip_fresh_process(tmp_path):
    p, target = fixture()
    p = Problem(p.data, target[:, None], p.row_ids, weight=p.weight)
    model = squared(p, p, context=RunContext("categorical", 3), rounds=3, bins=4).state.model
    path = tmp_path / "model.json"
    model.save(path)
    restored = Model.load(path)
    assert restored.identity == model.identity
    code = """import json, sys
from openboost import MixedData
from openboost.artifacts import Model
x = MixedData([[0, 'a'], [100, 'b'], [None, 'unseen']], [1, 2, 3], ('x','c'), ('numeric','categorical'))
print(json.dumps(Model.load(sys.argv[1]).predict(x).tolist()))
"""
    result = subprocess.check_output([sys.executable, "-c", code, str(path)], text=True)
    x = MixedData(
        [[0, "a"], [100, "b"], [None, "unseen"]], [1, 2, 3], ("x", "c"), ("numeric", "categorical")
    )
    np.testing.assert_array_equal(json.loads(result), model.predict(x))
    for _kind, bad in (
        ("duplicate", ["a", "a", "c"]),
        ("unsorted", ["c", "b", "a"]),
        ("mixed", ["a", 1]),
        ("empty", []),
    ):
        record = model.record()
        record["terms"][0]["learner"]["categories"][1] = bad
        path.write_text(json.dumps(record))
        with pytest.raises(ValueError):
            Model.load(path)


def test_foreign_kinds_dictionary_and_invalid_tokens_rejected():
    p, gradient = fixture()
    b = Binning.fit(p.data).transform(p.data)
    tree = depthwise(b, newton(p, gradient, np.ones(8)))
    numeric = NumericData(np.zeros((8, 2)), p.row_ids, p.data.feature_names)
    with pytest.raises(ValueError):
        tree.predict(numeric)
    candidate = choose(candidates(histogram(b, newton(p, gradient, np.ones(8)))))
    other = Binning(
        b.binning.feature_names, b.binning.cuts, (None, ("a", "b", "c", "d"))
    ).transform(p.data)
    with pytest.raises(ValueError):
        partition(other, None, candidate)
    for tokens in ([True, False], [1, "a"], [1.5, 2.5]):
        with pytest.raises(ValueError):
            MixedData([[v] for v in tokens], [1, 2], ("c",), ("categorical",))
    record = tree.record()
    record["categories"][1] = None
    with pytest.raises(ValueError):
        Tree.from_record(record)


def test_all_missing_and_missing_only_category_split(tmp_path):
    data = MixedData(
        [["a", None], ["a", None], [None, None], [None, None]],
        [1, 2, 3, 4],
        ("c", "empty"),
        ("categorical", "categorical"),
    )
    p = Problem(data, [[0]] * 4, data.row_ids)
    b = Binning.fit(data).transform(data)
    options = candidates(histogram(b, newton(p, [-2, -2, 2, 2], [1] * 4)))
    assert options and all(c.feature == 0 for c in options)
    tree = depthwise(b, newton(p, [-2, -2, 2, 2], [1] * 4))
    path = tmp_path / "tree.json"
    tree.save(path)
    unseen = MixedData(
        [["a", "new"], ["new", None]], [10, 11], data.feature_names, data.feature_kinds
    )
    assert Tree.load(path).predict(unseen)[0, 0] != tree.predict(unseen)[1, 0]
    with pytest.raises(ValueError):
        Binning(("c",), ([],), ("abc",))
