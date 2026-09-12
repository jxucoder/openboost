"""Production scalar comparison expressions against the independent 111 oracle."""

from decimal import Decimal

import numpy as np
import pytest

from openboost._comparison_math import cpu_add, cpu_div, cpu_mul
from openboost._multiclass_comparison_math import cpu_row_change
from openboost.comparison import _multiclass_result

from .reference.multiclass_comparison import CASES, direct_difference


def evaluate(case):
    old, new, target, offset, weight = (np.asarray(a, np.float32) for a in case["arrays"])
    total, mass, code = (0.0, 0.0), (0.0, 0.0), 0
    for a, b, y, off, w in zip(old, new, target, offset, weight, strict=True):
        lower, upper, status = cpu_row_change(a, b, int(y), off)
        total = cpu_add(total, cpu_mul((lower, upper), (float(w), float(w))))
        mass = cpu_add(mass, (float(w), float(w)))
        code = max(code, status)
    return _multiclass_result(*cpu_div(total, mass), code, np.array_equal(old, new))


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_production_expression_encloses_independent_decimal(case):
    result = evaluate(case)
    exact = direct_difference(*case["arrays"])
    assert result.lower is not None
    assert Decimal(result.lower) <= exact <= Decimal(result.upper)
    if case["status"] is not None:
        assert result.status == case["status"]
    assert result.unchanged == (case["arrays"][0] == case["arrays"][1])
    if result.improves():
        assert exact < 0
        assert not result.improves(abs(result.lower) * 2)


@pytest.mark.parametrize(
    "raw,new,expected",
    [([0, 513], [0, 514], 1), ([0, 0], [0, 300], 1), ([1e308, -1e308], [0, 0], 2)],
)
def test_private_expression_range_is_explicit(raw, new, expected):
    assert cpu_row_change(np.asarray(raw), np.asarray(new), 0, np.zeros(2))[2] == expected
