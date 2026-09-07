"""Private bounded Normal algebra with explicit backend rounding dependencies.

Only basic scalar operations are backend-specific. No platform exp/expm1 is used
for an enclosure. See the 092-A derivation; this is not a general interval API.
The independent original-row Decimal oracle lives outside this module.
"""

import math


def make_normal_math(compile_function, add_down, add_up, mul_down, mul_up, div_down, div_up):
    """Build Python or CUDA scalar operations at import time, never by host fallback."""
    zero, one, two, unknown = (0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (-math.inf, math.inf)

    @compile_function
    def finite(a):
        return math.isfinite(a[0]) and math.isfinite(a[1])

    @compile_function
    def add(a, b):
        if not finite(a) or not finite(b):
            return unknown
        if a == zero:
            return b
        if b == zero:
            return a
        result = add_down(a[0], b[0]), add_up(a[1], b[1])
        return result if finite(result) else unknown

    @compile_function
    def mul(a, b):
        if not finite(a) or not finite(b):
            return unknown
        if a == zero or b == zero:
            return zero
        if a == one:
            return b
        if b == one:
            return a
        lo, hi = math.inf, -math.inf
        for x in a:
            for y in b:
                low, high = mul_down(x, y), mul_up(x, y)
                if not math.isfinite(low) or not math.isfinite(high):
                    return unknown
                lo, hi = min(lo, low), max(hi, high)
        return lo, hi

    @compile_function
    def div(a, b):
        if not finite(a) or not finite(b) or b[0] <= 0 <= b[1]:
            return unknown
        if a == zero:
            return zero
        if b == one:
            return a
        lo, hi = math.inf, -math.inf
        for x in a:
            for y in b:
                low, high = div_down(x, y), div_up(x, y)
                if not math.isfinite(low) or not math.isfinite(high):
                    return unknown
                lo, hi = min(lo, low), max(hi, high)
        return lo, hi

    @compile_function
    def exp_point(x, minus_one):
        if x == 0:
            return zero if minus_one else one
        t, halvings = (x, x), 0
        while max(abs(t[0]), abs(t[1])) > 0.0625:
            t = div(t, two)
            halvings += 1
        term, total = t, t
        for degree in range(2, 19):
            term = div(mul(term, t), (float(degree), float(degree)))
            total = add(total, term)
        absolute_term = max(abs(term[0]), abs(term[1]))
        absolute_t = max(abs(t[0]), abs(t[1]))
        tail = div(
            mul(mul(two, (absolute_term, absolute_term)), (absolute_t, absolute_t)), (19.0, 19.0)
        )[1]
        result = add(total, (-tail, tail))
        if not minus_one:
            result = add(one, result)
        for _ in range(halvings):
            result = mul(result, add(result, two)) if minus_one else mul(result, result)
        return result

    @compile_function
    def exp_bound(a, minus_one):
        # Callers check this support before use, including arithmetic overflow.
        return exp_point(a[0], minus_one)[0], exp_point(a[1], minus_one)[1]

    @compile_function
    def row_change(m0, l0, m1, l1, y, mo, lo):
        old_r = add(add((m0, m0), (mo, mo)), (-y, -y))
        new_r = add(add((m1, m1), (mo, mo)), (-y, -y))
        old_l, new_l = add((l0, l0), (lo, lo)), add((l1, l1), (lo, lo))
        dm = zero if m0 == m1 else add((m1, m1), (-m0, -m0))
        dl = zero if l0 == l1 else add((l1, l1), (-l0, -l0))
        a, b, c = mul((-2.0, -2.0), old_l), mul((-2.0, -2.0), new_l), mul((-2.0, -2.0), dl)
        for exponent in (a, b, c):
            if not finite(exponent):
                return 0.0, 0.0, 2
            if exponent[0] < -64 or exponent[1] > 64:
                return 0.0, 0.0, 1
        old_p, new_p = exp_bound(a, False), exp_bound(b, False)
        # These checks also prevent zero weight from hiding an overflowing row.
        for r, p, ell in ((old_r, old_p, old_l), (new_r, new_p, new_l)):
            if not finite(add(div(mul(mul(r, r), p), two), ell)):
                return 0.0, 0.0, 2
        polynomial = add(mul(mul(two, old_r), dm), mul(dm, dm))
        change = add(
            dl,
            mul(
                div(old_p, two),
                add(
                    mul(polynomial, exp_bound(c, False)),
                    mul(mul(old_r, old_r), exp_bound(c, True)),
                ),
            ),
        )
        if not finite(change):
            return 0.0, 0.0, 2
        return change[0], change[1], 0

    return add, mul, div, row_change


def _add_down(a, b):
    return math.nextafter(a + b, -math.inf)


def _add_up(a, b):
    return math.nextafter(a + b, math.inf)


def _mul_down(a, b):
    return math.nextafter(a * b, -math.inf)


def _mul_up(a, b):
    return math.nextafter(a * b, math.inf)


def _div_down(a, b):
    return math.nextafter(a / b, -math.inf)


def _div_up(a, b):
    return math.nextafter(a / b, math.inf)


cpu_add, cpu_mul, cpu_div, cpu_row_change = make_normal_math(
    lambda function: function, _add_down, _add_up, _mul_down, _mul_up, _div_down, _div_up
)
