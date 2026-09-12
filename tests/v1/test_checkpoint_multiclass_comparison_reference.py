"""Self-contained checkpoint controls extracted from tests/v1/test_multiclass_comparison_reference.py.

Source: 91a519dd5227344266a3eb1c86ce21b45acf32c6. Original test/helper bodies are retained;
archive-only trajectory/study replay remains in the original full evidence checkout.
Extraction is not a new numerical validation result.
"""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from openboost import ClassSchema, NumericData, Problem
from openboost.objectives import Multiclass

from .reference.multiclass_comparison import CASES, compare, direct_difference, exp_bound
from .reference.normal_comparison import Interval


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_preregistered_softmax_bound_contains_direct_likelihood(case):
    result = compare(*case["arrays"])
    exact = direct_difference(*case["arrays"])
    alternate = direct_difference(*case["arrays"], precision=180)
    assert abs(exact - alternate) <= max(Decimal("1e-150"), abs(exact) * Decimal("1e-80"))
    assert result.lower is not None
    assert Decimal(result.lower) <= exact <= Decimal(result.upper)
    if case["status"] is not None:
        assert result.status == case["status"]
    old, new, *rest = case["arrays"]
    reverse = compare(new, old, *rest)
    assert Decimal(reverse.lower) <= -exact <= Decimal(reverse.upper)


@pytest.mark.parametrize("value", [-512, -100, -80, -1e-30, 0, 1e-30, 80, 512])
def test_exponential_encloses_direct_decimal(value):
    bound = exp_bound(Interval.point(value))
    with localcontext() as context:
        context.prec = 180
        exact = Decimal(value).exp()
    assert Decimal(bound.lower) <= exact <= Decimal(bound.upper)


@pytest.mark.parametrize("value", [1e-10, 1e-30])
def test_stationary_reporting_tie_is_proven_worsening(value):
    data = NumericData([[0]], [0], ("x",))
    p = Problem(data, [[0]], [0], raw_width=3, classes=ClassSchema(("a", "b", "c")))
    old, new = np.zeros((1, 3), np.float32), np.array([[0, value, -value]], np.float32)
    assert Multiclass.loss(p, old) == Multiclass.loss(p, new)
    result = compare(old, new, [0], p.offset, p.weight)
    assert result.status == "worsening" and result.lower > 0


@pytest.mark.parametrize("fault", ["shape", "nonfinite", "bad-zero-weight", "invalid-target"])
def test_all_row_validation_precedes_identity(fault):
    raw = np.zeros((2, 3))
    target = [0, 1]
    if fault == "nonfinite":
        raw[-1, 0] = np.inf
    elif fault == "bad-zero-weight":
        raw[-1, 0] = 120
    elif fault == "invalid-target":
        target[-1] = 3
    with pytest.raises((ValueError, FloatingPointError)):
        compare(raw, raw[:, :2] if fault == "shape" else raw, target, np.zeros_like(raw), [1, 0])


def test_class_and_row_permutation_keep_valid_enclosure():
    case = next(c for c in CASES if c["id"] == "weighted-offsets")
    old, new, target, offset, weight = (np.asarray(a) for a in case["arrays"])
    order, rows = np.array([2, 0, 1]), np.array([2, 0, 1])
    exact = direct_difference(*case["arrays"])
    result = compare(
        old[rows][:, order],
        new[rows][:, order],
        np.argsort(order)[target[rows].astype(int)],
        offset[rows][:, order],
        weight[rows],
    )
    assert Decimal(result.lower) <= exact <= Decimal(result.upper)


def test_scope_contains_unique_preregistered_settings():
    assert len(CASES) == len({c["id"] for c in CASES}) == 64
    assert {len(c["arrays"][0][0]) for c in CASES} == {2, 3, 5}
