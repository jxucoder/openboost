"""117 independent component controls, executable without CUDA packages."""

from functools import partial

import numpy as np
import pytest

from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.ops import candidates, histogram, vector_feasible, vector_leaf, vector_score
from openboost.stats import vector_newton
from openboost.tree import depthwise

from .reference import device_vector as ref


def prepared(width=2, projected=False, case="weighted"):
    source = ref.fixture(width, projected, case)
    data = NumericData(source["x"], 101 + 7 * np.arange(len(source["x"])), ("a", "b"))
    problem = Problem(data, np.zeros((len(data.row_ids), 1)), data.row_ids, weight=source["weight"])
    binned = Binning(data.feature_names, ([0.5, 1.5], [0.5])).transform(data)
    split = vector_newton(problem, source["split_g"], source["split_h"])
    leaves = vector_newton(problem, source["g"], source["h"])
    return source, problem, binned, split, leaves


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("rows", [tuple(range(8)), (6, 0, 4, 7), (3,)])
@pytest.mark.parametrize("minimum", [0, 2])
def test_exact_original_row_candidates(width, rows, minimum):
    source, _, binned, fields, _ = prepared(width)
    expected = ref.candidates(
        source["x"],
        source["g"] * source["weight"][:, None],
        source["h"] * source["weight"][:, None],
        rows,
        regularization=2,
        penalty=0.5,
        minimum=minimum,
    )
    actual = candidates(histogram(binned, fields, rows))
    assert [c.key for c in actual] == [c["key"] for c in expected]
    for candidate, known in zip(actual, expected, strict=True):
        for side, sums in enumerate((candidate.left, candidate.right)):
            np.testing.assert_array_equal(sums, [*known["g"][side], *known["h"][side]])
        assert vector_feasible(candidate, min_child_h=minimum) == known["legal"]
        if known["legal"]:
            assert vector_score(candidate, reg_lambda=2, split_penalty=0.5) == pytest.approx(
                float(known["gain"]), abs=1e-12
            )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_cpu_tree_matches_exact_vector_oracle(width, projected, depth):
    source, _, binned, fields, leaves = prepared(width, projected)
    known = ref.tree(source, depth, regularization=2, penalty=0.5, minimum=1)
    actual = depthwise(
        binned,
        fields,
        max_depth=depth,
        leaf_fields=leaves,
        leaf=partial(vector_leaf, reg_lambda=2),
        scoring=partial(vector_score, reg_lambda=2, split_penalty=0.5),
        legality=partial(vector_feasible, min_child_h=1),
    )
    assert actual.output_width == width
    assert [
        (None if f == -1 else (f, t, m))
        for f, t, m in zip(actual.feature, actual.threshold, actual.missing_left, strict=True)
    ] == [n["key"] for n in known]
    np.testing.assert_array_equal(actual.left, [n["left"] for n in known])
    np.testing.assert_array_equal(actual.right, [n["right"] for n in known])
    np.testing.assert_allclose(
        actual.value, [[float(v) for v in n["value"]] for n in known], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        actual.predict(binned.data), ref.predict(known, source["x"]), rtol=0, atol=1e-12
    )


def test_hand_tie_penalty_and_projection_change():
    source = ref.fixture(2, case="ties")
    options = ref.candidates(source["x"], source["g"], source["h"], range(8))
    chosen = ref.winner(options)
    assert chosen["key"] == (0, 0, False)
    assert float(chosen["gain"]) == pytest.approx(25.6)
    penalized = ref.candidates(source["x"], source["g"], source["h"], range(8), penalty=3)
    assert ref.winner(penalized)["gain"] == chosen["gain"] - 3  # once, not per channel
    assert ref.tree(ref.fixture(2), 1)[0]["key"] != ref.tree(ref.fixture(2, True), 1)[0]["key"]
    zero = ref.fixture(2, case="zero_channel")
    assert ref.tree(zero, 2)[0]["key"] is None
    with pytest.raises(ValueError):
        ref.leaf((1, 2), (1, 0), 0)
