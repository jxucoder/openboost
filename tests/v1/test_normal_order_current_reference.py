"""Current stored-field conformance fixture and corrected public CPU consumer."""

from fractions import Fraction

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost.binning import Binning
from openboost.newton_order import leaf, rank
from openboost.stats import RowFields
from openboost.tree import depthwise

from .reference.normal_split_order import captured, exact_tree, predict


def prepared(case):
    x, exact = captured(stored=True)
    if case != "captured":
        power, direction = case.split("-")
        mass = Fraction(1, 2 ** int(power[1:]))
        exact[0] = ({"negative": -1, "zero": 0, "positive": 1}[direction] * mass, mass)
    values = [[np.nan if v is None else v for v in row] for row in x]
    data = NumericData(values, 101 + 7 * np.arange(8), ("f0", "f1"))
    problem = Problem(data, np.zeros((8, 1)), data.row_ids, weight=[float(h) for _, h in exact])
    binned = Binning(data.feature_names, (np.arange(3) + 0.5, np.arange(3) + 0.5)).transform(data)
    stored = np.asarray(exact, np.float32)
    assert [[Fraction(float(v)) for v in row] for row in stored] == [list(row) for row in exact]
    fields = RowFields(problem.identity, data.identity, ("gradient", "curvature"), stored,
                       ("training", "training"))
    return problem, binned, fields, x, exact


def topology(nodes):
    return [(-1, -1, False, n["left"], n["right"]) if n["key"] is None
            else (*n["key"], n["left"], n["right"]) for n in nodes]


@pytest.mark.parametrize("case", [
    "captured", "p24-negative", "p24-zero", "p24-positive",
    "p54-negative", "p54-zero", "p54-positive",
    "p100-negative", "p100-zero", "p100-positive",
    "p149-negative", "p149-zero", "p149-positive",
])
def test_corrected_cpu_exact_order_and_original_row_leaves(case):
    problem, binned, fields, x, exact = prepared(case)
    wanted = exact_tree(x, exact, depth=2)
    tree = depthwise(binned, fields, max_depth=2, ordering=rank, field_leaf=leaf)
    actual = list(zip(tree.feature, tree.threshold, tree.missing_left, tree.left, tree.right, strict=True))
    assert actual == topology(wanted)
    np.testing.assert_allclose(tree.value[:, 0], [float(n["value"]) for n in wanted], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(tree.predict(problem.data)[:, 0], [float(v) for v in predict(wanted, x)],
                               rtol=1e-14, atol=1e-14)
