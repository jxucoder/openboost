"""118 CPU mathematical controls established before resident implementation."""

from fractions import Fraction
from functools import partial

import numpy as np
import pytest

from openboost.binning import Binning
from openboost.data import NumericData, Problem
from openboost.objectives import MultiSquared
from openboost.ops import vector_feasible, vector_leaf, vector_score
from openboost.stats import vector_newton
from openboost.tree import depthwise

from .reference import multi_squared as ref


def prepared(width=2, conflict=False):
    f = ref.fixture(width, conflict)
    problems = []
    for prefix, start in (("", 101), ("validation_", 501)):
        data = NumericData(f[prefix + "x"], start + 7 * np.arange(8), ("a", "b"))
        problems.append(
            Problem(
                data,
                f[prefix + "target"],
                data.row_ids,
                weight=f[prefix + "weight"],
                offset=f[prefix + "offset"],
                raw_width=width,
            )
        )
    binning = Binning(("a", "b"), ([0.5, 1.5], [0.5]))
    return *problems, binning


@pytest.mark.parametrize("case", ref.CASES, ids=lambda c: c["id"])
def test_exact_polynomial_equals_direct_loss_difference(case):
    old, new, *rest = case["arrays"]
    exact = ref.change(*case["arrays"])
    assert exact == ref.loss(new, *rest) - ref.loss(old, *rest)
    assert ref.change(new, old, *rest) == -exact
    rows, columns = np.arange(len(old))[::-1], np.arange(old.shape[1])[::-1]
    y, off, weight = rest
    assert (
        ref.change(
            old[rows][:, columns],
            new[rows][:, columns],
            y[rows][:, columns],
            off[rows][:, columns],
            weight[rows],
        )
        == exact
    )


@pytest.mark.parametrize("width", [1, 2, 4])
@pytest.mark.parametrize("mode", ["independent", "shared", "projected"])
@pytest.mark.parametrize("depth", [0, 1, 2])
def test_two_round_cpu_geometry_and_tree_composition(width, mode, depth):
    train, validation, binning = prepared(width)
    base, steps = ref.rounds(width, mode, depth)
    np.testing.assert_allclose(MultiSquared.base(train), base, rtol=1e-6, atol=1e-6)
    raw = np.tile(base, (8, 1))
    for step in steps:
        np.testing.assert_allclose(
            MultiSquared.gradient(train, raw), step["gradient"], rtol=1e-6, atol=1e-6
        )
        for channel, (nodes, mapping) in enumerate(step["terms"]):
            g = step["gradient"]
            if mode == "independent":
                g = g[:, channel : channel + 1]
            leaves = vector_newton(train, g, np.ones_like(g))
            fields = (
                vector_newton(train, g[:, :1], np.ones_like(g[:, :1]))
                if mode == "projected"
                else leaves
            )
            tree = depthwise(
                binning.transform(train.data),
                fields,
                max_depth=depth,
                leaf_fields=leaves,
                scoring=vector_score,
                legality=vector_feasible,
                leaf=partial(vector_leaf, reg_lambda=1),
            )
            assert [
                (None if f == -1 else (f, t, m))
                for f, t, m in zip(tree.feature, tree.threshold, tree.missing_left, strict=True)
            ] == [n["key"] for n in nodes]
            np.testing.assert_allclose(
                tree.value, [[float(v) for v in n["value"]] for n in nodes], atol=1e-12
            )
            raw = ref.mapped(
                raw, tree.predict(train.data).astype(np.float32), mapping, np.float32(0.5)
            )
        np.testing.assert_array_equal(raw, step["raw"])
        assert MultiSquared.loss(train, raw) == pytest.approx(
            float(ref.loss(raw, train.target, train.offset, train.weight))
        )


def test_reporting_ties_hide_strict_worsening_and_mapping_rounds_independently():
    for case in ref.CASES[-3:]:
        old, new, target, offset, weight = case["arrays"]
        data = NumericData([[0]], [0], ("x",))
        problem = Problem(data, target, [0], offset=offset, weight=weight, raw_width=2)
        assert MultiSquared.loss(problem, old) == MultiSquared.loss(problem, new)
        assert ref.change(*case["arrays"]) > 0
    # An exact halfway plus a smaller-than-binary64 perturbation must round up.
    midpoint = Fraction(1) + Fraction(1, 2**24)
    assert ref.stored(midpoint) == np.float32(1)
    assert ref.stored(midpoint + Fraction(1, 2**80)) == np.nextafter(np.float32(1), np.float32(2))
    raw = np.zeros((1, 1), np.float32)
    prediction = np.array([[2**24, 1, -(2**24)]], np.float32)
    mapping = np.ones((3, 1), np.float32)
    assert ref.mapped(raw, prediction, mapping, np.float32(1))[0, 0] == 0
    assert (prediction.astype(float) @ mapping.astype(float)).item() == 1
