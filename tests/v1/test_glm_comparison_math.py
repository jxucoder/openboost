"""Production scalar algebra tested as Python against the frozen independent oracle."""

from decimal import Decimal

import pytest

from openboost._comparison_math import cpu_add, cpu_div, cpu_mul
from openboost._glm_comparison_math import cpu_row_change
from openboost.comparison import _glm_result

from .reference.glm_comparison import CASES, direct_difference


def evaluate(case):
    total, mass, code = (0.0, 0.0), (0.0, 0.0), 0
    old, new, y, off, weight, exposure = case["arrays"]
    for a, b, target, offset, w, e in zip(old, new, y, off, weight, exposure, strict=True):
        lo, hi, row_code = cpu_row_change(case["family"] == "binary", a, b, target, offset, e)
        total = cpu_add(total, cpu_mul((lo, hi), (w, w)))
        mass = cpu_add(mass, (w, w))
        code = max(code, row_code)
    return _glm_result(case["family"], *cpu_div(total, mass), code, old == new)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_production_expression_contains_independent_likelihood(case):
    result = evaluate(case)
    exact = direct_difference(case["family"], *case["arrays"])
    assert Decimal(result.lower) <= exact <= Decimal(result.upper)
    if case["status"]:
        assert result.status == case["status"]
    assert result.unchanged == (case["arrays"][0] == case["arrays"][1])
    if result.improves():
        assert exact < 0
        assert not result.improves(abs(result.lower) * 2)


@pytest.mark.parametrize("binary", [True, False])
def test_private_expression_range_is_explicit(binary):
    assert cpu_row_change(binary, 256, 257, 1, 0, 1)[2] == 1
    assert cpu_row_change(binary, 1e308, -1e308, 1, 1e308, 1)[2] == 2
