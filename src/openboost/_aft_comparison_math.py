"""Enclosing AFT event polynomial and censored integral; see Sprint 115-A."""

import math

from ._comparison_math import cpu_add, cpu_div, cpu_mul


def make_aft_math(compile_function, add, mul, div):
    """Build bounded scalar expressions with explicit CPU/CUDA rounding operations."""
    zero, one, two, half = (0., 0.), (1., 1.), (2., 2.), (.5, .5)
    log_two = (0.6931471805599453, 0.6931471805599454)
    density_constant = (0.39894228040143265, 0.3989422804014327)

    @compile_function
    def negative(a):
        return -a[1], -a[0]

    @compile_function
    def finite(a):
        return math.isfinite(a[0]) and math.isfinite(a[1])

    @compile_function
    def logarithm(value):
        if value == 1:
            return zero
        # Power-of-two multiplication is exact here for positive float32 inputs.
        m, exponent = value, 0
        while m < 1:
            m *= 2
            exponent -= 1
        while m >= 2:
            m *= 0.5
            exponent += 1
        point = m, m
        t = div(add(point, (-1., -1.)), add(point, one))
        square, power, total = mul(t, t), t, zero
        for k in range(32):
            denominator = float(2*k+1)
            total = add(total, div(power, (denominator, denominator)))
            power = mul(power, square)
        remainder = div(mul(two, power), mul((65., 65.), add(one, negative(square))))[1]
        return add(add(mul(log_two, (float(exponent), float(exponent))), mul(two, total)), (0., remainder))

    @compile_function
    def exponential(point):
        if point == 0:
            return one
        t, halvings = (point, point), 0
        while max(abs(t[0]), abs(t[1])) > .0625:
            t = div(t, two)
            halvings += 1
        term, total = t, t
        for degree in range(2, 19):
            term = div(mul(term, t), (float(degree), float(degree)))
            total = add(total, term)
        a, b = max(abs(term[0]), abs(term[1])), max(abs(t[0]), abs(t[1]))
        remainder = div(mul(mul(two, (a, a)), (b, b)), (19., 19.))[1]
        result = add(one, add(total, (-remainder, remainder)))
        for _ in range(halvings):
            result = mul(result, result)
        return result

    @compile_function
    def density(exponent):
        bound = exponential(exponent[0])[0], exponential(exponent[1])[1]
        return mul(density_constant, bound)

    @compile_function
    def mills(point):
        x = abs(point), abs(point)
        if abs(point) > 2:
            correction = 0., div((129., 129.), x)[1]
            for k in range(128, 0, -1):
                correction = div((float(k), float(k)), add(x, correction))
            positive = add(x, correction)
            if point > 0:
                return positive
            phi = density(negative(div(mul(x, x), two)))
            return div(phi, add(one, negative(div(phi, positive))))
        z = point, point
        power, integral = z, zero
        square = negative(div(mul(z, z), two))
        for k in range(32):
            denominator = float(2*k+1)
            integral = add(integral, div(power, (denominator, denominator)))
            power = div(mul(power, square), (float(k+1), float(k+1)))
        absolute = max(abs(power[0]), abs(power[1]))
        remainder = div((absolute, absolute), (65., 65.))[1]
        integral = add(integral, (-remainder, remainder))
        return div(density(square), add(half, negative(mul(density_constant, integral))))

    @compile_function
    def row_change(old, new, lower, event, offset, sigma):
        # The caller validates both full geometry domains before this shortcut.
        if old == new:
            return 0., 0., 0
        scale = sigma, sigma
        delta = div(add((new, new), (-old, -old)), scale)
        log_time = logarithm(lower)
        before = div(add(add(log_time, (-old, -old)), (-offset, -offset)), scale)
        if not finite(delta) or not finite(before):
            return 0., 0., 2
        if event:
            change = mul(delta, add(negative(before), div(delta, two)))
        else:
            after = div(add(add(log_time, (-new, -new)), (-offset, -offset)), scale)
            if not finite(after):
                return 0., 0., 2
            left, right = min(before[0], after[0]), max(before[1], after[1])
            if left < -16 or right > 1e12:
                return 0., 0., 1
            rectangle = mul(negative(delta), (mills(left)[0], mills(right)[1]))
            gradient = negative((mills(before[0])[0], mills(before[1])[1]))
            remainder = div(mul(delta, delta), two)[1]
            taylor = add(mul(gradient, delta), (0., remainder))
            change = max(rectangle[0], taylor[0]), min(rectangle[1], taylor[1])
        if not finite(change) or change[0] > change[1]:
            return 0., 0., 2
        return change[0], change[1], 0

    return logarithm, mills, row_change


cpu_logarithm, cpu_mills, cpu_row_change = make_aft_math(lambda f: f, cpu_add, cpu_mul, cpu_div)
