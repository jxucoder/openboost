"""GLM row enclosures and ordered reduction on the owning CUDA stream."""

import math

from numba import cuda, float64
from numba.cuda import libdevice

from ._comparison_math import make_normal_math
from ._device_glm_kernels import binary_row, poisson_row
from ._glm_comparison_math import make_glm_math

_add, _mul, _div, _unused_normal = make_normal_math(
    cuda.jit(device=True),
    libdevice.dadd_rd,
    libdevice.dadd_ru,
    libdevice.dmul_rd,
    libdevice.dmul_ru,
    libdevice.ddiv_rd,
    libdevice.ddiv_ru,
)
_change = make_glm_math(cuda.jit(device=True), _add, _mul, _div)


@cuda.jit
def binary_compare_rows(target, offset, before, after, output):
    r = cuda.grid(1)
    if r < before.shape[0]:
        old, new = float64(before[r, 0]), float64(after[r, 0])
        y, off = float64(target[r, 0]), float64(offset[r, 0])
        old_loss, _, _ = binary_row(old + off, y)
        new_loss, _, _ = binary_row(new + off, y)
        lower, upper, code = float64(0), float64(0), 3
        if math.isfinite(old_loss) and math.isfinite(new_loss):
            lower, upper, code = _change(True, old, new, y, off, float64(1))
        output[r, 0], output[r, 1], output[r, 2] = lower, upper, code
        output[r, 3] = 1 if old == new else 0


@cuda.jit
def poisson_compare_rows(target, offset, exposure, before, after, output):
    r = cuda.grid(1)
    if r < before.shape[0]:
        old, new = float64(before[r, 0]), float64(after[r, 0])
        y, off, e = float64(target[r, 0]), float64(offset[r, 0]), float64(exposure[r, 0])
        old_loss, _, _ = poisson_row(old + off, y, e)
        new_loss, _, _ = poisson_row(new + off, y, e)
        lower, upper, code = float64(0), float64(0), 3
        if math.isfinite(old_loss) and math.isfinite(new_loss):
            lower, upper, code = _change(False, old, new, y, off, e)
        output[r, 0], output[r, 1], output[r, 2] = lower, upper, code
        output[r, 3] = 1 if old == new else 0


@cuda.jit
def glm_compare_reduce(rows, weight, output):
    if cuda.grid(1) == 0:
        total, mass = (float64(0), float64(0)), (float64(0), float64(0))
        code, unchanged = 0, 1
        for r in range(rows.shape[0]):
            code = max(code, int(rows[r, 2]))
            if rows[r, 3] == 0:
                unchanged = 0
            w = float64(weight[r]), float64(weight[r])
            total = _add(total, _mul((rows[r, 0], rows[r, 1]), w))
            mass = _add(mass, w)
        lower, upper = _div(total, mass)
        if not math.isfinite(lower) or not math.isfinite(upper):
            code = max(code, 2)
        output[0], output[1], output[2], output[3] = lower, upper, code, unchanged
