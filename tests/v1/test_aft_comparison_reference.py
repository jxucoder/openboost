"""115-A independent AFT bounds precede production comparison construction."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from .reference.aft_comparison import (
    CASES,
    DENSITY,
    LOG_TWO,
    compare,
    direct_difference,
    logarithm,
    mills,
)
from .reference.device_aft import DOMAIN_CASES, geometry, pi, tail


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_independent_enclosure_contains_direct_likelihood_and_reverse(case):
    arrays = case["arrays"]
    low = direct_difference(*arrays, precision=160)
    exact = direct_difference(*arrays, precision=220)
    assert abs(low-exact) <= max(Decimal("1e-140"), abs(exact)*Decimal("1e-50"))
    result = compare(*arrays)
    assert result.lower is not None
    assert Decimal(result.lower) <= exact <= Decimal(result.upper)
    if case["status"]:
        assert result.status == case["status"]
    reverse = compare(arrays[1], arrays[0], *arrays[2:])
    assert Decimal(reverse.lower) <= -exact <= Decimal(reverse.upper)
    permuted = compare(*(a[::-1] for a in arrays[:-1]), arrays[-1])
    assert Decimal(permuted.lower) <= exact <= Decimal(permuted.upper)


@pytest.mark.parametrize("x", [float(np.nextafter(np.float32(0), np.float32(1))), 1e-40, .1, .5, 1, 1.5, 2, 17, 1e38])
def test_logarithm_contains_independent_decimal(x):
    x = float(np.float32(x))
    with localcontext() as ctx:
        ctx.prec = 160
        exact = Decimal(x).ln()
    bound = logarithm(x)
    assert Decimal(bound.lower) <= exact <= Decimal(bound.upper)


@pytest.mark.parametrize("z", [-16, -14, -8, -2.001, -2, -1, 0, 1, 2, 2.001, 8, 1000, 1e8, 1e12])
def test_mills_contains_independent_decimal(z):
    exact = tail(z, precision=160)[1]
    bound = mills(z)
    assert Decimal(bound.lower) <= exact <= Decimal(bound.upper)


def test_constants_enclose_independent_decimal():
    with localcontext() as ctx:
        ctx.prec = 160
        assert Decimal(LOG_TWO.lower) < Decimal(2).ln() < Decimal(LOG_TWO.upper)
        assert Decimal(DENSITY.lower) < 1/(2*pi(160)).sqrt() < Decimal(DENSITY.upper)


@pytest.mark.parametrize("event", [False, True])
def test_rounded_reporting_tie_does_not_supply_sign(event):
    old, new = [0], [1e-20]
    assert geometry(old, [1], [event], [0], [1], 1)[0] == geometry(new, [1], [event], [0], [1], 1)[0]
    assert compare(old, new, [1], [event], [0], [1], 1).status == ("worsening" if event else "improvement")


@pytest.mark.parametrize("case", [c for c in DOMAIN_CASES if not c[-1]], ids=lambda c: c[0])
def test_invalid_identical_zero_weight_row_still_raises(case):
    _, raw, lower, event, sigma, _ = case
    with pytest.raises((ValueError, FloatingPointError)):
        compare([0, raw], [0, raw], [1, lower], [True, event], [0, 0], [1, 0], sigma)


def test_finite_tail_range_is_explicit_and_noop_is_validated():
    result = compare([-1e13], [-1e13+1e6], [1], [False], [0], [1], 1)
    assert result.status == "unresolved" and result.reason == "tail_range"
    assert compare([-1e13], [-1e13], [1], [False], [0], [1], 1).status == "unchanged"


def test_freeze_is_unique():
    assert len({c["id"] for c in CASES}) == len(CASES) == 62
