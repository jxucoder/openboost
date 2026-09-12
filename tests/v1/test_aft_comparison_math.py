"""Shared production AFT scalar expressions versus independent direct likelihoods."""

from decimal import Decimal, localcontext

import pytest

from openboost._aft_comparison_math import cpu_logarithm, cpu_mills, cpu_row_change
from openboost._comparison_math import cpu_add, cpu_div, cpu_mul
from openboost.comparison import _aft_result

from .reference.aft_comparison import CASES, direct_difference
from .reference.device_aft import tail


def evaluate(arrays):
    old, new, lower, event, offset, weight, sigma = arrays
    total, mass, code = (0., 0.), (0., 0.), 0
    for a, b, t, e, o, w in zip(old, new, lower, event, offset, weight, strict=True):
        lo, hi, status = cpu_row_change(a, b, t, e, o, sigma)
        total = cpu_add(total, cpu_mul((lo, hi), (w, w)))
        mass = cpu_add(mass, (w, w))
        code = max(code, status)
    return _aft_result(*cpu_div(total, mass), code, old == new)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_production_bounds_contain_direct_likelihood(case):
    arrays = case["arrays"]
    exact = direct_difference(*arrays)
    result = evaluate(arrays)
    assert Decimal(result.lower) <= exact <= Decimal(result.upper)
    if case["status"]:
        assert result.status == case["status"]
    reverse = evaluate([arrays[1], arrays[0], *arrays[2:]])
    assert Decimal(reverse.lower) <= -exact <= Decimal(reverse.upper)
    assert result.unchanged == (arrays[0] == arrays[1])
    if result.improves():
        assert exact < 0 and not result.improves(abs(result.lower)*2)


@pytest.mark.parametrize("x", [2**-149, 2**-126, .1, .5, 1, 1.5, 2, 17, 1e38])
def test_shared_log_expression_contains_decimal(x):
    with localcontext() as ctx:
        ctx.prec = 160
        exact = Decimal(x).ln()
    lo, hi = cpu_logarithm(x)
    assert Decimal(lo) <= exact <= Decimal(hi)


@pytest.mark.parametrize("z", [-16, -14, -8, -2.001, -2, -1, 0, 1, 2, 2.001, 8, 1000, 1e8, 1e12])
def test_shared_mills_expression_contains_decimal(z):
    lo, hi = cpu_mills(z)
    exact = tail(z, precision=160)[1]
    assert Decimal(lo) <= exact <= Decimal(hi)


def test_private_range_reasons_and_exact_noop():
    assert cpu_row_change(-1e13, -1e13+1e6, 1, False, 0, 1)[2] == 1
    assert cpu_row_change(1e308, -1e308, 1, True, 0, 1)[2] == 2
    assert cpu_row_change(-1e13, -1e13, 1, False, 0, 1) == (0, 0, 0)
