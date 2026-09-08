"""107-A numerical counterexamples precede production comparator construction."""

from decimal import Decimal

import numpy as np
import pytest

from .reference.device_glm import geometry
from .reference.glm_comparison import CASES, compare, direct_difference, exp_bound
from .reference.normal_comparison import Interval


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_frozen_convex_enclosure_and_direct_likelihood(case):
    result = compare(case["family"], *case["arrays"])
    low_precision = direct_difference(case["family"], *case["arrays"], precision=160)
    exact = direct_difference(case["family"], *case["arrays"], precision=220)
    assert abs(low_precision - exact) <= max(Decimal("1e-140"), abs(exact) * Decimal("1e-50"))
    assert result.lower is not None
    assert Decimal(result.lower) <= exact <= Decimal(result.upper)
    if case["status"]:
        assert result.status == case["status"]
    arrays = case["arrays"]
    reversed_result = compare(case["family"], arrays[1], arrays[0], *arrays[2:])
    assert Decimal(reversed_result.lower) <= -exact <= Decimal(reversed_result.upper)
    permuted = compare(case["family"], *(list(reversed(a)) for a in arrays))
    assert Decimal(permuted.lower) <= exact <= Decimal(permuted.upper)


@pytest.mark.parametrize("x", [-256, -100, -80, -1, -1e-30, 0, 1e-30, 1, 80, 256])
def test_extended_exponential_contains_decimal(x):
    from decimal import localcontext

    bound = exp_bound(Interval.point(x))
    with localcontext() as context:
        context.prec = 160
        exact = Decimal(x).exp()
    assert Decimal(bound.lower) <= exact <= Decimal(bound.upper)


def test_poisson_reporting_tie_is_mathematical_worsening():
    case = next(c for c in CASES if c["id"] == "poisson/stationary-rounded-tie")
    old, new, y, o, w, e = case["arrays"]
    assert geometry("poisson", new, y, o, w, e)[0] == geometry("poisson", old, y, o, w, e)[0]
    assert compare("poisson", *case["arrays"]).status == "worsening"


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_identical_invalid_zero_weight_row_cannot_be_unchanged(family):
    raw = [0, 120 if family == "binary" else 90]
    with pytest.raises((ValueError, FloatingPointError)):
        compare(family, raw, raw, [0, 1], [0, 0], [1, 0], [1, 1])


def test_freeze_has_both_families_and_no_silently_duplicate_cases():
    assert len({c["id"] for c in CASES}) == len(CASES) == 59
    assert {c["family"] for c in CASES} == {"binary", "poisson"}
    assert all(np.asarray(c["arrays"]).dtype.kind == "f" for c in CASES)
