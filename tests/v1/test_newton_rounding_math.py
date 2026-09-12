"""Integer ratio rounding checked against an independent exponent/quantum oracle."""

import numpy as np
import pytest

from .reference.binary32_rounding import bits, cases


def array(value, width=100):
    assert 0 <= value < 2**(16*width)
    return np.array([(value >> (16*i)) & 65535 for i in range(width)], np.uint32)


@pytest.mark.parametrize('case', ['zero', 'cancel', 'one', 'negative_one', 'half_even', 'half_above', 'half_below',
                                  'half_odd', 'negative_half_above', 'minimum', 'half_minimum', 'negative_half_minimum',
                                  'above_half_minimum', 'below_half_minimum', 'odd_subnormal_midpoint', 'smallest_normal',
                                  'normal_boundary', 'below_normal_boundary', 'maximum', 'below_overflow', 'overflow',
                                  'above_overflow', 'negative_below_overflow', 'negative_overflow'])
def test_frozen_ieee_rounding_oracle_has_no_float_intermediate(case):
    value, expected = cases()[case]
    if expected is None:
        with pytest.raises(OverflowError):
            bits(value)
    else:
        assert bits(value) == expected


@pytest.mark.parametrize('case', ['zero', 'cancel', 'one', 'negative_one', 'half_even', 'half_above', 'half_below',
                                  'half_odd', 'negative_half_above', 'minimum', 'half_minimum', 'negative_half_minimum',
                                  'above_half_minimum', 'below_half_minimum', 'odd_subnormal_midpoint', 'smallest_normal',
                                  'normal_boundary', 'below_normal_boundary', 'maximum', 'below_overflow', 'overflow',
                                  'above_overflow', 'negative_below_overflow', 'negative_overflow'])
@pytest.mark.parametrize('scale', [1, 17])
def test_fixed_width_ratio_rounding_matches_rational_oracle_and_ieee_bits(case, scale):
    from openboost._newton_round_math import make_rounding

    value, expected = cases()[case]
    operation = make_rounding(lambda function: function, np.uint64)
    n, d = abs(value.numerator)*scale, value.denominator*scale
    work = np.zeros((6, 100), np.uint32)
    result, invalid = operation(array(n), array(d), value < 0, work)
    if expected is None:
        assert invalid == 1
    else:
        assert invalid == 0 and result == bits(value) == expected


def test_zero_denominator_and_insufficient_capacity_are_explicit_failures():
    from openboost._newton_round_math import make_rounding

    operation = make_rounding(lambda function: function, np.uint64)
    assert operation(array(0), array(0), False, np.zeros((6, 100), np.uint32))[1] == 1
    assert operation(array(1), array(0), True, np.zeros((6, 100), np.uint32))[1] == 1
    assert operation(array(1, 1), array(1, 1), False, np.zeros((6, 1), np.uint32))[1] == 1
