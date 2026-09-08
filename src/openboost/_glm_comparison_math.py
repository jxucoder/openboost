"""Bounded convex GLM algebra with explicit rounding dependencies; see 107-A."""

import math

from ._comparison_math import cpu_add, cpu_div, cpu_mul


def make_glm_math(compile_function, add, mul, div):
    """Build scalar Python/CUDA expressions; neither libm bounds nor host fallback."""
    one, two = (1.0, 1.0), (2.0, 2.0)

    @compile_function
    def finite(a):
        return math.isfinite(a[0]) and math.isfinite(a[1])

    @compile_function
    def exp_point(x):
        if x == 0:
            return one
        t, halvings = (x, x), 0
        while max(abs(t[0]), abs(t[1])) > 0.0625:
            t = div(t, two)
            halvings += 1
        term, total = t, t
        for degree in range(2, 19):
            term = div(mul(term, t), (float(degree), float(degree)))
            total = add(total, term)
        a, b = max(abs(term[0]), abs(term[1])), max(abs(t[0]), abs(t[1]))
        tail = div(mul(mul(two, (a, a)), (b, b)), (19.0, 19.0))[1]
        result = add(one, add(total, (-tail, tail)))
        for _ in range(halvings):
            result = mul(result, result)
        return result

    @compile_function
    def exp_bound(a):
        return exp_point(a[0])[0], exp_point(a[1])[1]

    @compile_function
    def sigmoid(x):
        if x == 0:
            return 0.5, 0.5
        tail = exp_point(-abs(x))
        return div(one if x > 0 else tail, add(one, tail))

    @compile_function
    def curvature(distance):
        if distance == 0:
            return 0.25, 0.25
        tail = exp_point(-distance)
        denominator = add(one, tail)
        return div(tail, mul(denominator, denominator))

    @compile_function
    def row_change(binary, old, new, target, offset, exposure):
        # Caller validates both complete geometry domains before this exact identity.
        if old == new:
            return 0.0, 0.0, 0
        before = add((old, old), (offset, offset))
        after = add((new, new), (offset, offset))
        delta = add((new, new), (-old, -old))
        if binary and target == 1:
            before, after, delta = (
                (-before[1], -before[0]),
                (-after[1], -after[0]),
                (-delta[1], -delta[0]),
            )
        if not finite(before) or not finite(after) or not finite(delta):
            return 0.0, 0.0, 2
        left, right = min(before[0], after[0]), max(before[1], after[1])
        if left < -256 or right > 256:
            return 0.0, 0.0, 1
        if binary:
            gradient = sigmoid(before[0])[0], sigmoid(before[1])[1]
            far = max(abs(left), abs(right))
            near = 0.0 if left <= 0 <= right else min(abs(left), abs(right))
            h = curvature(far)[0], min(0.25, curvature(near)[1])
        else:
            e = exposure, exposure
            gradient = add(mul(e, exp_bound(before)), (-target, -target))
            h = mul(e, exp_bound((left, right)))
        change = add(mul(gradient, delta), div(mul(h, mul(delta, delta)), two))
        if not finite(change):
            return 0.0, 0.0, 2
        return change[0], change[1], 0

    return row_change


cpu_row_change = make_glm_math(lambda function: function, cpu_add, cpu_mul, cpu_div)
