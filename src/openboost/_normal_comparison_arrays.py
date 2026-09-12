"""Bounded CPU row batching with the original scalar arithmetic order.

This implements the same interval polynomial as _comparison_math. It does not
use platform exp/expm1 for an enclosure and does not reorder row accumulation.
"""

import numpy as np


def _aligned(a, b):
    return np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(b, np.float64))


def _finite(a):
    return np.isfinite(a).all(axis=-1)


def _finish(out, valid):
    out[~valid] = (-np.inf, np.inf)
    return out


def add(a, b):
    a, b = _aligned(a, b)
    with np.errstate(all="ignore"):
        out = np.stack(
            (np.nextafter(a[:, 0] + b[:, 0], -np.inf), np.nextafter(a[:, 1] + b[:, 1], np.inf)),
            axis=1,
        )
    # The scalar's a==zero branch precedes its b==zero branch, including -0.
    out = np.where((b == 0).all(axis=1)[:, None], a, out)
    out = np.where((a == 0).all(axis=1)[:, None], b, out)
    return _finish(out, _finite(a) & _finite(b) & _finite(out))


def _endpoint_operation(a, b, divide):
    a, b = _aligned(a, b)
    lo = np.full(len(a), np.inf)
    hi = -lo.copy()
    valid = _finite(a) & _finite(b)
    arithmetic = np.ones(len(a), bool)
    with np.errstate(all="ignore"):
        for i in (0, 1):
            for j in (0, 1):
                value = a[:, i] / b[:, j] if divide else a[:, i] * b[:, j]
                low, high = np.nextafter(value, -np.inf), np.nextafter(value, np.inf)
                arithmetic &= np.isfinite(low) & np.isfinite(high)
                # Python min/max retain their first operand on a tie, including
                # opposite signed zeros. NumPy minimum/maximum need not do so.
                lo = np.where(low < lo, low, lo)
                hi = np.where(high > hi, high, hi)
    out = np.stack((lo, hi), axis=1)
    azero = (a == 0).all(axis=1)
    bzero = (b == 0).all(axis=1)
    aone = (a == 1).all(axis=1)
    bone = (b == 1).all(axis=1)
    out = np.where(bone[:, None], a, out)
    if divide:
        out = np.where(azero[:, None], 0.0, out)
        valid &= ~((b[:, 0] <= 0) & (b[:, 1] >= 0))
        arithmetic |= azero | bone
    else:
        out = np.where(aone[:, None], b, out)
        out = np.where((azero | bzero)[:, None], 0.0, out)
        arithmetic |= azero | bzero | aone | bone
    return _finish(out, valid & arithmetic & _finite(out))


def mul(a, b):
    return _endpoint_operation(a, b, False)


def div(a, b):
    return _endpoint_operation(a, b, True)


def _point(values):
    return np.stack((values, values), axis=1)


def _exp_point(values, minus_one):
    if not np.any(values != 0):
        return _point(np.zeros_like(values) if minus_one else np.ones_like(values))
    t = _point(values)
    halvings = np.zeros(len(values), np.int32)
    active = np.max(np.abs(t), axis=1) > 0.0625
    while active.any():
        t[active] = div(t[active], (2.0, 2.0))
        halvings[active] += 1
        active = np.max(np.abs(t), axis=1) > 0.0625
    term, total = t.copy(), t.copy()
    for degree in range(2, 19):
        term = div(mul(term, t), (float(degree), float(degree)))
        total = add(total, term)
    absolute_term = _point(np.max(np.abs(term), axis=1))
    absolute_t = _point(np.max(np.abs(t), axis=1))
    tail = div(mul(mul((2.0, 2.0), absolute_term), absolute_t), (19.0, 19.0))[:, 1]
    result = add(total, np.stack((-tail, tail), axis=1))
    if not minus_one:
        result = add((1.0, 1.0), result)
    for level in range(int(halvings.max(initial=0))):
        active = halvings > level
        current = result[active]
        result[active] = (
            mul(current, add(current, (2.0, 2.0))) if minus_one else mul(current, current)
        )
    result[values == 0] = (0.0, 0.0) if minus_one else (1.0, 1.0)
    return result


def _exp_bound(a, minus_one=False):
    endpoints = _exp_point(a.reshape(-1), minus_one).reshape(len(a), 2, 2)
    return np.stack((endpoints[:, 0, 0], endpoints[:, 1, 1]), axis=1)


def _chunk(before, after, target, offset):
    m0, l0 = before.T
    m1, l1 = after.T
    mo, lo = offset.T
    old_r = add(add(_point(m0), _point(mo)), _point(-target))
    new_r = add(add(_point(m1), _point(mo)), _point(-target))
    old_l, new_l = add(_point(l0), _point(lo)), add(_point(l1), _point(lo))
    dm = add(_point(m1), _point(-m0))
    dm[m0 == m1] = 0.0
    dl = add(_point(l1), _point(-l0))
    dl[l0 == l1] = 0.0
    a, b, c = (mul((-2.0, -2.0), v) for v in (old_l, new_l, dl))
    code = np.zeros(len(before), np.int32)
    for exponent in (a, b, c):
        code[(code == 0) & ~_finite(exponent)] = 2
        code[(code == 0) & ((exponent[:, 0] < -64) | (exponent[:, 1] > 64))] = 1
    # Preserve first-failure priority. Invalid rows never enter exponent loops.
    safe = [np.where((code == 0)[:, None], v, 0.0) for v in (a, b, c)]
    old_p, new_p = _exp_bound(safe[0]), _exp_bound(safe[1])
    for residual, precision, ell in ((old_r, old_p, old_l), (new_r, new_p, new_l)):
        domain = add(div(mul(mul(residual, residual), precision), (2.0, 2.0)), ell)
        code[(code == 0) & ~_finite(domain)] = 2
    polynomial = add(mul(mul((2.0, 2.0), old_r), dm), mul(dm, dm))
    change = add(
        dl,
        mul(
            div(old_p, (2.0, 2.0)),
            add(
                mul(polynomial, _exp_bound(safe[2])),
                mul(mul(old_r, old_r), _exp_bound(safe[2], True)),
            ),
        ),
    )
    code[(code == 0) & ~_finite(change)] = 2
    change[code != 0] = 0.0
    return np.column_stack((change, code))


def rows(before, after, target, offset):
    """Evaluate aligned binary64 rows; public domain checks precede this helper."""
    before, after, target, offset = (
        np.asarray(v, np.float64) for v in (before, after, target, offset)
    )
    result = np.empty((len(before), 3), np.float64)
    for start in range(0, len(before), 4096):
        stop = start + 4096
        result[start:stop] = _chunk(
            before[start:stop], after[start:stop], target[start:stop], offset[start:stop]
        )
    return result
