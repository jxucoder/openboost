"""Softmax path-variance bounds with explicit rounding dependencies; see 111-A."""

import math

from ._comparison_math import cpu_add, cpu_div, cpu_mul


def make_multiclass_math(compile_function, add, mul, div, cast=float):
    """Build scalar expressions for Python checking and resident CUDA."""
    zero, one, two = (0.0, 0.0), (1.0, 1.0), (2.0, 2.0)

    @compile_function
    def finite(value):
        return math.isfinite(value[0]) and math.isfinite(value[1])

    @compile_function
    def sum_bound(left, right):
        if left[0] == left[1] and right[0] == right[1] and left[0] == -right[0]:
            return zero
        return add(left, right)

    @compile_function
    def difference(left, right):
        return zero if left == right else add((left, left), (-right, -right))

    @compile_function
    def square(value):
        product = mul(value, value)
        return max(0.0, product[0]), product[1]

    @compile_function
    def exp_point(value):
        if value == 0:
            return one
        reduced, halvings = (value, value), 0
        while max(abs(reduced[0]), abs(reduced[1])) > 0.0625:
            reduced = div(reduced, two)
            halvings += 1
        term, total = reduced, reduced
        for degree in range(2, 19):
            term = div(mul(term, reduced), (float(degree), float(degree)))
            total = add(total, term)
        a, b = max(abs(term[0]), abs(term[1])), max(abs(reduced[0]), abs(reduced[1]))
        remainder = div(mul(mul(two, (a, a)), (b, b)), (19.0, 19.0))[1]
        result = add(one, add(total, (-remainder, remainder)))
        for _ in range(halvings):
            result = mul(result, result)
        return result

    @compile_function
    def exp_bound(value):
        return exp_point(value[0])[0], exp_point(value[1])[1]

    @compile_function
    def parts(before, after, target, offset, j):
        old_relative = difference(cast(before[j]), cast(before[target]))
        new_relative = difference(cast(after[j]), cast(after[target]))
        effective = sum_bound(old_relative, difference(cast(offset[j]), cast(offset[target])))
        delta = sum_bound(new_relative, (-old_relative[1], -old_relative[0]))
        if not finite(effective) or not finite(delta):
            return zero, zero, 2
        if effective[0] < -512 or effective[1] > 512:
            return zero, zero, 1
        return exp_bound(effective), delta, 0

    @compile_function
    def row_change(before, after, target, offset):
        # Caller validates both complete geometry domains before this identity.
        unchanged = True
        for j in range(before.size):
            unchanged = unchanged and before[j] == after[j]
        if unchanged:
            return 0.0, 0.0, 0
        mass, linear, variance = zero, zero, zero
        low_step, high_step = 0.0, 0.0
        for i in range(before.size):
            a, delta, code = parts(before, after, target, offset, i)
            if code:
                return 0.0, 0.0, code
            mass = add(mass, a)
            linear = sum_bound(linear, mul(a, delta))
            low_step, high_step = min(low_step, delta[0]), max(high_step, delta[1])
            for j in range(i):
                b, other, code = parts(before, after, target, offset, j)
                if code:
                    return 0.0, 0.0, code
                gap = sum_bound(delta, (-other[1], -other[0]))
                variance = add(variance, mul(mul(a, b), square(gap)))
        linear, variance = div(linear, mass), div(variance, square(mass))
        radius = max(0.0, difference(high_step, low_step)[1])
        exponent = mul(two, (radius, radius))
        if not finite(linear) or not finite(variance) or not finite(exponent):
            return 0.0, 0.0, 2
        if exponent[1] > 512:
            return 0.0, 0.0, 1
        low = mul(exp_bound((-exponent[1], -exponent[0])), variance)
        high = mul(exp_bound(exponent), variance)
        global_upper = div(square((radius, radius)), (4.0, 4.0))[1]
        curvature = max(0.0, low[0]), min(high[1], global_upper)
        result = add(linear, div(curvature, two))
        if not finite(result) or curvature[0] > curvature[1]:
            return 0.0, 0.0, 2
        return result[0], result[1], 0

    return row_change


cpu_row_change = make_multiclass_math(lambda function: function, cpu_add, cpu_mul, cpu_div)
