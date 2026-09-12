"""Exact scalar Newton reference kernels; no host numerical refinement."""

from numba import cuda, uint32, uint64
from numba.cuda import libdevice

from ._newton_integer_math import make_math

_zero, _compare, _add, _subtract, _multiply, _decode, _score, _positive, _order = make_math(cuda.jit(device=True), uint64)


@cuda.jit(device=True)
def _magnitude(positive, negative, out):
    if _compare(positive, negative) >= 0:
        return _subtract(positive, negative, out)
    return _subtract(negative, positive, out)


@cuda.jit
def exact_newton_rank(fields, codes, missing, rows, active, width, g, h, regularization,
                      minimum, penalty, info, info_minima, sums, numerator, denominator, eligible, flags):
    c = cuda.grid(1)
    if c >= active.size:
        return
    eligible[c], flags[c] = False, 0
    _zero(numerator[c])
    _zero(denominator[c])
    for q in range(6):
        _zero(sums[c, q])
    if not active[c]:
        return
    # Compile-time storage sizes follow the independently checked exponent/row
    # proof. Every helper still checks actual output capacity before publication.
    temp = cuda.local.array((3, 20), uint32)
    values = cuda.local.array((6, 20), uint32)  # |Gl|, |Gr|, |Gp|, Dl, Dr, Dp.
    parameters = cuda.local.array((3, 20), uint32)
    work = cuda.local.array((4, 100), uint32)
    _, invalid = _decode(libdevice.float_as_int(regularization), parameters[0])
    _, failure = _decode(libdevice.float_as_int(minimum), parameters[1])
    invalid |= failure
    _, failure = _decode(libdevice.float_as_int(penalty), parameters[2])
    invalid |= failure
    f, threshold, missing_left = c//(2*width), (c//2) % width, c % 2 == 1
    nl, nr = 0, 0
    for j in range(rows.size):
        r = rows[j]
        left = missing_left if missing[f, r] else codes[f, r] <= threshold
        offset = 0 if left else 3
        if left:
            nl += 1
        else:
            nr += 1
        sign, failure = _decode(libdevice.float_as_int(fields[r, g]), temp[0])
        invalid |= failure
        q = offset if sign >= 0 else offset+1
        invalid |= _add(sums[c, q], temp[0], sums[c, q])
        sign, failure = _decode(libdevice.float_as_int(fields[r, h]), temp[0])
        invalid |= failure
        if sign < 0:
            invalid = 1
        invalid |= _add(sums[c, offset+2], temp[0], sums[c, offset+2])
    invalid |= _magnitude(sums[c, 0], sums[c, 1], values[0])
    invalid |= _magnitude(sums[c, 3], sums[c, 4], values[1])
    invalid |= _add(sums[c, 0], sums[c, 3], temp[0])
    invalid |= _add(sums[c, 1], sums[c, 4], temp[1])
    invalid |= _magnitude(temp[0], temp[1], values[2])
    invalid |= _add(sums[c, 2], parameters[0], values[3])
    invalid |= _add(sums[c, 5], parameters[0], values[4])
    invalid |= _add(sums[c, 2], sums[c, 5], temp[0])
    invalid |= _add(temp[0], parameters[0], values[5])
    _zero(temp[0])
    legal = (nl > 0 and nr > 0 and _compare(sums[c, 2], temp[0]) > 0
             and _compare(sums[c, 5], temp[0]) > 0
             and _compare(sums[c, 2], parameters[1]) >= 0
             and _compare(sums[c, 5], parameters[1]) >= 0)
    # Additional information is accumulated from its original stored column;
    # objective weights are never reapplied to these independent masses.
    for q in range(info.size):
        _zero(temp[1])
        _zero(temp[2])
        for j in range(rows.size):
            r = rows[j]
            left = missing_left if missing[f, r] else codes[f, r] <= threshold
            sign, failure = _decode(libdevice.float_as_int(fields[r, info[q]]), temp[0])
            invalid |= failure
            if sign < 0:
                invalid = 1
            side = 1 if left else 2
            invalid |= _add(temp[side], temp[0], temp[side])
        _, failure = _decode(libdevice.float_as_int(info_minima[q]), temp[0])
        invalid |= failure
        if _compare(temp[1], temp[0]) < 0 or _compare(temp[2], temp[0]) < 0:
            legal = False
    if legal:
        invalid |= _score(values[0], values[1], values[3], values[4], numerator[c], denominator[c], work)
        positive, failure = _positive(numerator[c], denominator[c], values[2], values[5], parameters[2], work)
        invalid |= failure
        eligible[c] = positive and invalid == 0
    flags[c] = invalid


@cuda.jit(device=True)
def _best(numerator, denominator, left, right):
    # -1 is an empty range; -2 is a checked arithmetic failure. Failures must
    # survive every later merge, even when the other range contains no candidate.
    if left == -2 or right == -2:
        return -2
    if left == -1:
        return right
    if right == -1:
        return left
    work = cuda.local.array((2, 100), uint32)
    comparison, invalid = _order(numerator[left], denominator[left], numerator[right], denominator[right], work)
    if invalid:
        return -2
    if comparison > 0 or (comparison == 0 and left < right):
        return left
    return right


@cuda.jit
def exact_newton_choose(numerator, denominator, eligible, extra, output):
    pair = cuda.grid(1)
    if pair >= output.size:
        return
    left, right = 2*pair, 2*pair+1
    if left >= eligible.size or not eligible[left] or not extra[left]:
        left = -1
    if right >= eligible.size or not eligible[right] or not extra[right]:
        right = -1
    output[pair] = _best(numerator, denominator, left, right)


@cuda.jit
def exact_newton_reduce(numerator, denominator, previous, output):
    pair = cuda.grid(1)
    if pair >= output.size:
        return
    left = previous[2*pair]
    right = previous[2*pair+1] if 2*pair+1 < previous.size else -1
    output[pair] = _best(numerator, denominator, left, right)
