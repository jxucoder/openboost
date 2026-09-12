"""Original-row Newton leaves checked by independent Fraction reductions."""

from dataclasses import replace
from fractions import Fraction as F
from functools import partial

import numpy as np
import pytest

from openboost import NumericData
from openboost.binning import Binning
from openboost.newton_order import leaf, rank
from openboost.stats import RowFields
from openboost.tree import best_first, depthwise, symmetric


def fixture(g=(1e16, 1, -1e16), h=(1, 1, 1)):
    data = NumericData(np.arange(len(g))[:, None], np.arange(len(g)), ("x",))
    f = RowFields(
        "p",
        data.identity,
        ("gradient", "curvature"),
        np.column_stack((g, h)),
        ("training", "training"),
    )
    return Binning.fit(data).transform(data), f


@pytest.mark.parametrize(
    "g,h,regularization",
    [
        ((1e16, 1, -1e16), (1, 1, 1), 1),
        ((1e308, 1e308, -1e308, -1e308, 1), (1, 1, 1, 1, 1), 2.5),
        ((1e308, 1e308), (1e308, 1e308), 0),
        ((1e-300,), (1e-300,), 0),
        ((0,), (0,), 1),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_exact_leaf_rounds_only_the_final_solution(g, h, regularization, reverse):
    b, f = fixture(g, h)
    rows = list(reversed(range(len(g)))) if reverse else list(range(len(g)))
    expected = -sum((F(float(g[i])) for i in rows), F(0)) / (
        sum((F(float(h[i])) for i in rows), F(0)) + F(regularization)
    )
    assert leaf(f, rows, reg_lambda=regularization) == float(expected)


@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
def test_field_leaf_preserves_cancellation_and_never_uses_floating_aggregation(grow, monkeypatch):
    b, f = fixture()
    from openboost import ops

    original = ops.histogram

    def guard(data, fields, rows=None):
        assert rows is not None and len(rows) == 0
        return original(data, fields, rows)

    monkeypatch.setattr(ops, "histogram", guard)
    tree = grow(b, f, max_depth=0, ordering=rank, field_leaf=leaf)
    np.testing.assert_array_equal(tree.value, [[-0.25]])


@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
def test_every_original_node_uses_exact_leaf_and_preserves_separate_field_ownership(grow):
    b, f = fixture((-4, -2, 1, 3), (1, 1, 1, 1))
    fields = replace(f, names=("curvature", "gradient"), values=f.values[:, ::-1])
    seen = []

    def field_leaf(actual, rows):
        assert actual is fields and not rows.flags.writeable
        seen.append(tuple(rows))
        return [leaf(actual, rows, reg_lambda=2), len(rows)]

    tree = grow(
        b, f, ordering=partial(rank, reg_lambda=2), field_leaf=field_leaf, leaf_fields=fields
    )
    assert len(seen) == len(tree.value) and tree.output_width == 2
    for i, rows in enumerate(seen):
        g = sum((F(float(f.values[r, 0])) for r in rows), F(0))
        h = sum((F(float(f.values[r, 1])) for r in rows), F(0))
        assert tree.value[i, 0] == float(-g / (h + 2)) and tree.value[i, 1] == len(rows)


@pytest.mark.parametrize(
    "fault", ["rows", "role", "curvature", "name", "lambda", "denominator", "overflow", "type"]
)
def test_leaf_domain_failures_are_explicit(fault):
    b, f = fixture()
    rows = None
    options = {}
    if fault == "rows":
        rows = [0, 0]
    elif fault == "role":
        f = replace(f, roles=("unweighted", "unweighted"))
    elif fault == "curvature":
        f = replace(f, values=f.values * np.array([1, -1]))
    elif fault == "name":
        f = replace(f, names=("g", "h"))
    elif fault == "lambda":
        options["reg_lambda"] = -1
    elif fault == "denominator":
        f = replace(f, values=np.zeros_like(f.values))
        options["reg_lambda"] = 0
    elif fault == "overflow":
        b, f = fixture((1e308,), (1e-308,))
        options["reg_lambda"] = 0
    else:
        f = None
    with pytest.raises(ValueError):
        leaf(f, rows, **options)


@pytest.mark.parametrize("grow", [depthwise, best_first, symmetric])
@pytest.mark.parametrize(
    "fault", ["callback", "additive", "residual", "foreign", "nonfinite", "empty"]
)
def test_field_leaf_conflicts_and_bad_outputs_fail_before_a_tree(grow, fault):
    b, f = fixture()
    kwargs = dict(field_leaf=leaf)
    if fault == "callback":
        kwargs["field_leaf"] = False
    elif fault == "additive":
        kwargs["leaf"] = lambda *_: 0
    elif fault == "residual":
        kwargs["row_leaf"] = lambda *_: 0
    elif fault == "foreign":
        kwargs["leaf_fields"] = replace(f, data_identity="foreign")
    elif fault == "nonfinite":
        kwargs["field_leaf"] = lambda *_: np.nan
    else:
        kwargs["field_leaf"] = lambda *_: []
    with pytest.raises(ValueError):
        grow(b, f, max_depth=0, **kwargs)
