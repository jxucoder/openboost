"""AFT row bounds with directed double arithmetic on the owning CUDA stream."""

import math

from numba import cuda, float64
from numba.cuda import libdevice

from ._aft_comparison_math import make_aft_math
from ._comparison_math import make_normal_math
from ._device_aft_kernels import aft_row

_add, _mul, _div, _unused_normal = make_normal_math(
    cuda.jit(device=True), libdevice.dadd_rd, libdevice.dadd_ru,
    libdevice.dmul_rd, libdevice.dmul_ru, libdevice.ddiv_rd, libdevice.ddiv_ru,
)
_log, _mills, _change = make_aft_math(cuda.jit(device=True), _add, _mul, _div)


@cuda.jit
def aft_compare_rows(lower, offset, event, sigma, before, after, output):
    r = cuda.grid(1)
    if r < before.shape[0]:
        old, new = float64(before[r, 0]), float64(after[r, 0])
        t, off, e = float64(lower[r, 0]), float64(offset[r, 0]), event[r]
        old_loss, _, _ = aft_row(old+off, t, e, sigma)
        new_loss, _, _ = aft_row(new+off, t, e, sigma)
        lo, hi, code = float64(0), float64(0), 3
        if math.isfinite(old_loss) and math.isfinite(new_loss):
            lo, hi, code = _change(old, new, t, e, off, sigma)
        output[r, 0], output[r, 1], output[r, 2] = lo, hi, code
        output[r, 3] = 1 if old == new else 0
