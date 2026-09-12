"""Resident multiclass comparison rows using directed binary64 arithmetic."""

import math

from numba import cuda, float64
from numba.cuda import libdevice

from ._comparison_math import make_normal_math
from ._device_multiclass_kernels import channel, partition
from ._multiclass_comparison_math import make_multiclass_math

_add, _mul, _div, _unused_normal = make_normal_math(
    cuda.jit(device=True),
    libdevice.dadd_rd,
    libdevice.dadd_ru,
    libdevice.dmul_rd,
    libdevice.dmul_ru,
    libdevice.ddiv_rd,
    libdevice.ddiv_ru,
)
_change = make_multiclass_math(cuda.jit(device=True), _add, _mul, _div, float64)


@cuda.jit
def multiclass_compare_rows(target, offset, before, after, output):
    r = cuda.grid(1)
    if r < before.shape[0]:
        old_max, old_index, old_tail = partition(before, offset, r)
        new_max, new_index, new_tail = partition(after, offset, r)
        valid, unchanged = True, True
        for j in range(before.shape[1]):
            old_g, old_h = channel(before, offset, r, j, target[r, 0], old_max, old_index, old_tail)
            new_g, new_h = channel(after, offset, r, j, target[r, 0], new_max, new_index, new_tail)
            valid = (
                valid
                and math.isfinite(old_g)
                and math.isfinite(old_h)
                and math.isfinite(new_g)
                and math.isfinite(new_h)
            )
            unchanged = unchanged and before[r, j] == after[r, j]
        lower, upper, code = float64(0), float64(0), 3
        if valid:
            lower, upper, code = _change(before[r], after[r], int(target[r, 0]), offset[r])
        output[r, 0], output[r, 1], output[r, 2] = lower, upper, code
        output[r, 3] = 1 if unchanged else 0
