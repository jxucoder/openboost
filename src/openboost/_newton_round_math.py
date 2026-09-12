"""Allocation-free exact ratio-to-binary32 rounding with explicit compilation.

The caller supplies six independent unsigned base-2^16 work arrays. Integer
bracketing and midpoint comparisons avoid a floating intermediate. Nonfinite
rounded results, zero denominators or capacity loss return an invalid flag.
"""

from ._newton_integer_math import make_math


def make_rounding(compile_function, wide):
    zero, compare, add, _, multiply, decode, _, _, _ = make_math(compile_function, wide)

    @compile_function
    def round_ratio(numerator, denominator, negative, work):
        zero(work[5])
        if compare(denominator, work[5]) == 0:
            return 0, 1
        if compare(numerator, work[5]) == 0:
            return 0, 0
        # One in binary32 is 2^149 integer units. Compare N*S to D*value_units.
        _, invalid = decode(1065353216, work[1])
        invalid |= multiply(numerator, work[1], work[0])
        low, high = 0, 2139095039
        while low < high:
            middle = (low+high+1)//2
            _, failure = decode(middle, work[1])
            invalid |= failure
            invalid |= multiply(work[1], denominator, work[2])
            if compare(work[2], work[0]) <= 0:
                low = middle
            else:
                high = middle-1
        _, failure = decode(low, work[3])
        invalid |= failure
        upper = low+1
        if upper == 2139095040:
            # Virtual 2^128 defines the finite-to-infinity rounding midpoint.
            _, failure = decode(2130706432, work[4])  # 2^127.
            invalid |= failure
            invalid |= add(work[4], work[4], work[4])
        else:
            _, failure = decode(upper, work[4])
            invalid |= failure
        invalid |= add(work[3], work[4], work[1])
        invalid |= multiply(work[1], denominator, work[2])
        invalid |= add(work[0], work[0], work[1])
        side = compare(work[1], work[2])
        result = upper if side > 0 or (side == 0 and low % 2 == 1) else low
        if invalid or result == 2139095040:
            return 0, 1
        return (result | 2147483648) if negative else result, 0

    return round_ratio
