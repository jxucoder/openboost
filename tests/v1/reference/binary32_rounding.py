"""Independent rational-to-binary32 rounding by exponent/quantum normalization.

No production imports, floating intermediate, binary search or limb arithmetic.
"""

from fractions import Fraction as F


def power(exponent):
    return F(2**exponent) if exponent >= 0 else F(1, 2**(-exponent))


def nearest_even(value):
    quotient, remainder = divmod(value.numerator, value.denominator)
    twice = 2*remainder
    return quotient + int(twice > value.denominator or (twice == value.denominator and quotient % 2 == 1))


def bits(value):
    value = F(value)
    if value == 0:
        return 0
    sign, magnitude = (0x80000000 if value < 0 else 0), abs(value)
    if magnitude < power(-126):
        return sign | nearest_even(magnitude / power(-149))
    exponent = magnitude.numerator.bit_length()-magnitude.denominator.bit_length()
    if magnitude < power(exponent):
        exponent -= 1
    significand = nearest_even(magnitude / power(exponent-23))
    if significand == 2**24:
        exponent, significand = exponent+1, 2**23
    if exponent > 127:
        raise OverflowError('exact rational rounds to nonfinite binary32')
    return sign | ((exponent+127) << 23) | (significand-2**23)


def cases():
    halfway = 1+power(-24)
    boundary = power(-126)-power(-150)
    maximum = F((2**24-1)*2**104)
    overflow = power(128)-power(103)
    return dict(
        zero=(F(0), 0), cancel=(F(-1, 4), 0xbe800000),
        one=(F(1), 0x3f800000), negative_one=(F(-1), 0xbf800000),
        half_even=(halfway, 0x3f800000), half_above=(halfway+power(-100), 0x3f800001),
        half_below=(halfway-power(-100), 0x3f800000),
        half_odd=(1+3*power(-24), 0x3f800002),
        negative_half_above=(-(halfway+power(-100)), 0xbf800001),
        minimum=(power(-149), 1), half_minimum=(power(-150), 0),
        negative_half_minimum=(-power(-150), 0x80000000),
        above_half_minimum=(power(-150)+power(-200), 1),
        below_half_minimum=(power(-150)-power(-200), 0),
        odd_subnormal_midpoint=(3*power(-150), 2),
        smallest_normal=(power(-126), 0x800000),
        normal_boundary=(boundary, 0x800000),
        below_normal_boundary=(boundary-power(-200), 0x7fffff),
        maximum=(maximum, 0x7f7fffff), below_overflow=(overflow-1, 0x7f7fffff),
        overflow=(overflow, None), above_overflow=(overflow+1, None),
        negative_below_overflow=(-(overflow-1), 0xff7fffff), negative_overflow=(-overflow, None),
    )
