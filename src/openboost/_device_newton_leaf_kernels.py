"""Original-row dyadic reduction and one correctly rounded binary32 leaf."""

from numba import cuda, int32, uint32, uint64
from numba.cuda import libdevice

from ._newton_integer_math import make_math
from ._newton_round_math import make_rounding

_zero, _compare, _add, _subtract, _, _decode, _, _, _ = make_math(cuda.jit(device=True), uint64)
_round = make_rounding(cuda.jit(device=True), uint64)


@cuda.jit
def exact_newton_leaf(fields, rows, g, h, regularization, output, status):
    if cuda.grid(1) != 0:
        return
    sums = cuda.local.array((3, 20), uint32)  # Positive G, negative G, H.
    values = cuda.local.array((3, 20), uint32)  # Scratch, |G|, H+lambda.
    work = cuda.local.array((6, 100), uint32)
    for q in range(3):
        _zero(sums[q])
    invalid = 0
    for j in range(rows.size):
        r = rows[j]
        sign, failure = _decode(libdevice.float_as_int(fields[r, g]), values[0])
        invalid |= failure
        q = 1 if sign < 0 else 0
        invalid |= _add(sums[q], values[0], sums[q])
        sign, failure = _decode(libdevice.float_as_int(fields[r, h]), values[0])
        invalid |= failure
        if sign < 0:
            invalid = 1
        invalid |= _add(sums[2], values[0], sums[2])
    negative = _compare(sums[0], sums[1]) > 0
    if negative:
        invalid |= _subtract(sums[0], sums[1], values[1])
    else:
        invalid |= _subtract(sums[1], sums[0], values[1])
    _, failure = _decode(libdevice.float_as_int(regularization), values[0])
    invalid |= failure
    invalid |= _add(sums[2], values[0], values[2])
    bits, failure = _round(values[1], values[2], negative, work)
    invalid |= failure
    status[0] = invalid
    output[0] = libdevice.int_as_float(int32(bits)) if invalid == 0 else 0.0
