"""Initial CUDA field/reduction kernels; invoked only on an owned context stream."""

import math

from numba import cuda, float32, float64
from numba.cuda import libdevice

from ._comparison_math import make_normal_math
from ._device_aft_comparison import aft_compare_rows as aft_compare_rows
from ._device_aft_kernels import aft_base as aft_base
from ._device_aft_kernels import aft_geometry as aft_geometry
from ._device_aft_kernels import aft_loss as aft_loss
from ._device_glm_comparison import binary_compare_rows as binary_compare_rows
from ._device_glm_comparison import glm_compare_reduce as glm_compare_reduce
from ._device_glm_comparison import poisson_compare_rows as poisson_compare_rows
from ._device_glm_kernels import (
    binary_base as binary_base,
)
from ._device_glm_kernels import (
    binary_geometry as binary_geometry,
)
from ._device_glm_kernels import (
    binary_loss as binary_loss,
)
from ._device_glm_kernels import (
    glm_gradient as glm_gradient,
)
from ._device_glm_kernels import (
    poisson_base as poisson_base,
)
from ._device_glm_kernels import (
    poisson_geometry as poisson_geometry,
)
from ._device_glm_kernels import (
    poisson_loss as poisson_loss,
)
from ._device_multi_squared_kernels import diagonal_fields as diagonal_fields
from ._device_multi_squared_kernels import multi_squared_base as multi_squared_base
from ._device_multi_squared_kernels import multi_squared_compare_rows as multi_squared_compare_rows
from ._device_multi_squared_kernels import multi_squared_geometry as multi_squared_geometry
from ._device_multi_squared_kernels import multi_squared_loss as multi_squared_loss
from ._device_multi_squared_kernels import projected_diagonal_fields as projected_diagonal_fields
from ._device_multiclass_comparison import multiclass_compare_rows as multiclass_compare_rows
from ._device_multiclass_kernels import multiclass_base as multiclass_base
from ._device_multiclass_kernels import multiclass_fields as multiclass_fields
from ._device_multiclass_kernels import multiclass_geometry as multiclass_geometry
from ._device_multiclass_kernels import multiclass_loss as multiclass_loss
from ._device_newton_kernels import exact_newton_choose as exact_newton_choose
from ._device_newton_kernels import exact_newton_rank as exact_newton_rank
from ._device_newton_kernels import exact_newton_reduce as exact_newton_reduce
from ._device_newton_leaf_kernels import exact_newton_leaf as exact_newton_leaf

_compare_add, _compare_mul, _compare_div, _normal_change = make_normal_math(
    cuda.jit(device=True), libdevice.dadd_rd, libdevice.dadd_ru,
    libdevice.dmul_rd, libdevice.dmul_ru, libdevice.ddiv_rd, libdevice.ddiv_ru,
)


@cuda.jit
def validate_fields(values, nonnegative, flags):
    # One cooperating block per field; no floating-point reduction or scratch.
    q = cuda.blockIdx.x
    invalid = 0
    if q < values.shape[1]:
        for r in range(cuda.threadIdx.x, values.shape[0], cuda.blockDim.x):
            v = values[r, q]
            if not math.isfinite(v) or (nonnegative and v < 0):
                invalid = 1
    # Every lane participates, including lanes with no rows in a partial tile.
    invalid = cuda.syncthreads_or(invalid)
    if q < values.shape[1] and cuda.threadIdx.x == 0:
        flags[q] = invalid


@cuda.jit
def weight_fields(values, weight, mask, output):
    i = cuda.grid(1)
    if i < values.size:
        r, q = i // values.shape[1], i % values.shape[1]
        output[r, q] = values[r, q] * weight[r] if mask[q] else values[r, q]


@cuda.jit
def append_field(values, column, output):
    i = cuda.grid(1)
    if i < output.size:
        r, q = i // output.shape[1], i % output.shape[1]
        output[r, q] = values[r, q] if q < values.shape[1] else column[r]


@cuda.jit
def validate_rows(rows, seen, flags):
    # Linear diagnostic validation; it never downloads the selected row array.
    if cuda.grid(1) == 0:
        flags[0] = 0
        for i in range(seen.size):
            seen[i] = 0
        for i in range(rows.size):
            r = rows[i]
            if r < 0 or r >= seen.size or seen[r] != 0:
                flags[0] = 1
            else:
                seen[r] = 1


@cuda.jit
def histogram(codes, missing, bins, values, rows, sums, counts):
    # One output cell per thread, original-row ordered sums, no floating atomics.
    i = cuda.grid(1)
    width, slots = values.shape[1], sums.shape[1]
    if i < sums.size:
        f, b, q = i // (slots * width), (i // width) % slots, i % width
        total, count = float32(0), 0
        if b <= bins[f]:
            for j in range(rows.size):
                r = rows[j]
                actual = bins[f] if missing[f, r] else codes[f, r]
                if actual == b:
                    total = float32(total + values[r, q])
                    count += 1
        sums[f, b, q] = total
        if q == 0:
            counts[f, b] = count


@cuda.jit
def row_total(values, rows, total):
    q = cuda.grid(1)
    if q < values.shape[1]:
        value = float32(0)
        for j in range(rows.size):
            value = float32(value + values[rows[j], q])
        total[q] = value


@cuda.jit
def candidate_sums(codes, missing, bins, sums, counts, output, child_counts, active):
    i = cuda.grid(1)
    width, slots = sums.shape[2], sums.shape[1] - 1
    if i < output.shape[0] * width:
        c, q = i // width, i % width
        f, threshold, missing_left = c // (2 * slots), (c // 2) % slots, c % 2 == 1
        left, right = float32(0), float32(0)
        nl, nr = 0, 0
        present = False
        if threshold < bins[f]:
            for b in range(threshold + 1):
                left = float32(left + sums[f, b, q])
                nl += counts[f, b]
            # Match CPU suffix addition order without parent-minus-child cancellation.
            for b in range(bins[f] - 1, threshold, -1):
                right = float32(right + sums[f, b, q])
                nr += counts[f, b]
            if missing_left:
                left = float32(left + sums[f, bins[f], q])
                nl += counts[f, bins[f]]
            else:
                right = float32(right + sums[f, bins[f], q])
                nr += counts[f, bins[f]]
            if q == 0:
                for r in range(codes.shape[1]):
                    if not missing[f, r] and codes[f, r] == threshold:
                        present = True
        output[c, 0, q], output[c, 1, q] = left, right
        if q == 0:
            active[c] = present
            child_counts[c, 0], child_counts[c, 1] = nl, nr


@cuda.jit
def scalar_scores(values, counts, active, parent, g, h, regularization, penalty, output):
    c = cuda.grid(1)
    if c < output.size:
        gain = float32(0)
        hl, hr = values[c, 0, h], values[c, 1, h]
        if active[c] and counts[c, 0] > 0 and counts[c, 1] > 0 and hl > 0 and hr > 0:
            dl, dr, dp = hl + regularization, hr + regularization, parent[h] + regularization
            if not math.isfinite(dl) or not math.isfinite(dr) or not math.isfinite(dp) or dp <= 0:
                gain = float32(math.nan)
            else:
                gl, gr = values[c, 0, g], values[c, 1, g]
                # Round products independently: contracting only one child into
                # the addition can break exact ties when children are swapped.
                left_score = libdevice.fmul_rn(float32(0.5) * gl, gl / dl)
                right_score = libdevice.fmul_rn(float32(0.5) * gr, gr / dr)
                parent_score = libdevice.fmul_rn(float32(0.5) * parent[g], parent[g] / dp)
                gain = (left_score + right_score) - parent_score - penalty
        output[c] = gain


@cuda.jit
def scalar_feasible(values, counts, active, h, minimum, output):
    c = cuda.grid(1)
    if c < output.size:
        hl, hr = values[c, 0, h], values[c, 1, h]
        output[c] = (
            active[c]
            and counts[c, 0] > 0
            and counts[c, 1] > 0
            and hl > 0
            and hr > 0
            and hl >= minimum
            and hr >= minimum
        )


@cuda.jit
def vector_scores(values, counts, active, parent, gradients, curvatures,
                  regularization, penalty, output):
    c = cuda.grid(1)
    if c < output.size:
        positive = active[c] and counts[c, 0] > 0 and counts[c, 1] > 0
        for k in range(len(curvatures)):
            h = curvatures[k]
            positive = positive and values[c, 0, h] > 0 and values[c, 1, h] > 0
        gain = float32(0)
        if positive:
            left_score, right_score, parent_score = float32(0), float32(0), float32(0)
            for k in range(len(gradients)):
                g, h = gradients[k], curvatures[k]
                dl = values[c, 0, h] + regularization
                dr = values[c, 1, h] + regularization
                dp = parent[h] + regularization
                if not math.isfinite(dl) or not math.isfinite(dr) or not math.isfinite(dp) or dp <= 0:
                    left_score = float32(math.nan)
                else:
                    gl, gr, gp = values[c, 0, g], values[c, 1, g], parent[g]
                    # Independent products preserve the scalar swapped-child symmetry.
                    left_score = float32(left_score + libdevice.fmul_rn(float32(0.5) * gl, gl / dl))
                    right_score = float32(right_score + libdevice.fmul_rn(float32(0.5) * gr, gr / dr))
                    parent_score = float32(parent_score + libdevice.fmul_rn(float32(0.5) * gp, gp / dp))
            gain = (left_score + right_score) - parent_score - penalty
        output[c] = gain


@cuda.jit
def vector_feasible(values, counts, active, curvatures, minimum, output):
    c = cuda.grid(1)
    if c < output.size:
        legal = active[c] and counts[c, 0] > 0 and counts[c, 1] > 0
        for k in range(len(curvatures)):
            h = curvatures[k]
            hl, hr = values[c, 0, h], values[c, 1, h]
            legal = legal and hl > 0 and hr > 0 and hl >= minimum and hr >= minimum
        output[c] = legal


@cuda.jit
def vector_leaf(total, gradients, curvatures, regularization, output):
    k = cuda.grid(1)
    if k < output.size:
        g, h = gradients[k], curvatures[k]
        denominator = total[h] + regularization
        output[k] = (
            -total[g] / denominator
            if total[h] >= 0 and denominator > 0 and math.isfinite(denominator)
            else float32(math.nan)
        )


@cuda.jit
def information_minimum(values, active, q, minimum, output):
    c = cuda.grid(1)
    if c < output.size:
        output[c] = active[c] and values[c, 0, q] >= minimum and values[c, 1, q] >= minimum


@cuda.jit
def combine_masks(left, right, output):
    c = cuda.grid(1)
    if c < output.size:
        output[c] = left[c] and right[c]


@cuda.jit
def choose_candidate(scores, mask, active, output):
    if cuda.grid(1) == 0:
        best, gain = -1, float32(0)
        for c in range(scores.size):
            if active[c] and mask[c] and scores[c] > gain:
                best, gain = c, scores[c]
        output[0] = best


@cuda.jit
def split_routes(codes, missing, rows, feature, threshold, missing_left, route, sizes):
    if cuda.grid(1) == 0:
        nl, nr = 0, 0
        for j in range(rows.size):
            r = rows[j]
            left = missing_left if missing[feature, r] else codes[feature, r] <= threshold
            route[j] = left
            if left:
                nl += 1
            else:
                nr += 1
        sizes[0], sizes[1] = nl, nr


@cuda.jit
def split_rows(rows, route, left, right):
    if cuda.grid(1) == 0:
        nl, nr = 0, 0
        for j in range(rows.size):
            if route[j]:
                left[nl] = rows[j]
                nl += 1
            else:
                right[nr] = rows[j]
                nr += 1


@cuda.jit
def scalar_leaf(total, g, h, regularization, output):
    if cuda.grid(1) == 0:
        denominator = total[h] + regularization
        output[0] = (
            -total[g] / denominator
            if total[h] >= 0 and denominator > 0 and math.isfinite(denominator)
            else float32(math.nan)
        )


@cuda.jit
def candidate_nonempty(counts, active, output):
    i = cuda.grid(1)
    if i < output.size:
        output[i] = active[i] and counts[i, 0] > 0 and counts[i, 1] > 0


@cuda.jit
def row_positions(output):
    i = cuda.grid(1)
    if i < output.size:
        output[i] = i


@cuda.jit
def squared_base(target, offset, weight, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(weight.size):
            total += float64(weight[r]) * (float64(target[r, 0]) - float64(offset[r, 0]))
            mass += float64(weight[r])
        output[0] = total / mass


@cuda.jit
def scalar_broadcast(value, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        output[r, 0] = value[0]


@cuda.jit
def squared_gradient(target, offset, raw, output):
    r = cuda.grid(1)
    if r < output.size:
        output[r] = float32(raw[r, 0] + offset[r, 0]) - target[r, 0]


@cuda.jit
def squared_fields(target, offset, raw, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        output[r, 0] = float32(raw[r, 0] + offset[r, 0]) - target[r, 0]
        output[r, 1] = float32(1)


@cuda.jit
def squared_loss(target, offset, weight, raw, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(weight.size):
            residual = float64(raw[r, 0]) + float64(offset[r, 0]) - float64(target[r, 0])
            total += float64(weight[r]) * residual * residual / 2
            mass += float64(weight[r])
        output[0] = total / mass


@cuda.jit
def pack_leaf(value, index, output):
    if cuda.grid(1) == 0:
        output[index] = value[0]


@cuda.jit
def scalar_tree_predict(codes, missing, topology, values, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        i = 0
        while topology[i, 0] != -1:
            f, t = topology[i, 0], topology[i, 1]
            left = topology[i, 2] != 0 if missing[f, r] else codes[f, r] <= t
            i = topology[i, 3] if left else topology[i, 4]
        output[r, 0] = values[i]


@cuda.jit
def pack_vector_leaf(value, index, output):
    k = cuda.grid(1)
    if k < value.size:
        output[index, k] = value[k]


@cuda.jit
def vector_tree_predict(codes, missing, topology, values, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        i = 0
        while topology[i, 0] != -1:
            f, t = topology[i, 0], topology[i, 1]
            left = topology[i, 2] != 0 if missing[f, r] else codes[f, r] <= t
            i = topology[i, 3] if left else topology[i, 4]
        for k in range(output.shape[1]):
            output[r, k] = values[i, k]


@cuda.jit
def scalar_add_raw(raw, delta, coefficient, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        output[r, 0] = raw[r, 0] + float32(coefficient * delta[r, 0])


@cuda.jit
def raw_broadcast(value, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        for k in range(output.shape[1]):
            output[r, k] = value[k]


@cuda.jit(device=True)
def normal_row(mean, ell, target):
    precision = math.exp(-2 * ell)
    scale = math.exp(ell)
    residual = mean - target
    square = residual * residual * precision
    g0, g1, h0 = float32(residual * precision), float32(1 - square), float32(precision)
    value = ell + square / 2 + math.log(2 * math.pi) / 2
    if (
        not math.isfinite(value)
        or not math.isfinite(g0)
        or not math.isfinite(g1)
        or not math.isfinite(h0)
        or h0 <= 0
        or not math.isfinite(float32(scale))
        or float32(scale) <= 0
    ):
        return float64(math.nan), float32(math.nan), float32(math.nan), float32(math.nan)
    return value, g0, g1, h0


@cuda.jit
def normal_base(target, offset, weight, floor, output):
    if cuda.grid(1) == 0:
        total, adjusted_mass, mass = float64(0), float64(0), float64(0)
        for r in range(weight.size):
            mass += float64(weight[r])
        for r in range(weight.size):
            precision = math.exp(-2 * float64(offset[r, 1]))
            if not math.isfinite(precision) or precision <= 0:
                output[0], output[1] = math.nan, math.nan
                return
            q = (float64(weight[r]) / mass) * precision
            total += q * (float64(target[r, 0]) - float64(offset[r, 0]))
            adjusted_mass += q
        if not math.isfinite(adjusted_mass) or adjusted_mass <= 0:
            output[0], output[1] = math.nan, math.nan
            return
        mean = total / adjusted_mass
        variance = float64(0)
        for r in range(weight.size):
            residual = float64(target[r, 0]) - float64(offset[r, 0]) - mean
            variance += (
                (float64(weight[r]) / mass)
                * math.exp(-2 * float64(offset[r, 1]))
                * residual
                * residual
            )
        output[0] = mean
        output[1] = math.log(max(math.sqrt(variance), float64(floor)))


@cuda.jit
def normal_geometry(target, offset, raw, gradient, fisher):
    r = cuda.grid(1)
    if r < raw.shape[0]:
        value, g0, g1, h0 = normal_row(
            float64(raw[r, 0]) + float64(offset[r, 0]),
            float64(raw[r, 1]) + float64(offset[r, 1]),
            float64(target[r, 0]),
        )
        gradient[r, 0], gradient[r, 1] = g0, g1
        fisher[r, 0], fisher[r, 1] = h0, float32(2)


@cuda.jit
def normal_loss(target, offset, weight, raw, output):
    if cuda.grid(1) == 0:
        total, mass = float64(0), float64(0)
        for r in range(weight.size):
            value, _, _, _ = normal_row(
                float64(raw[r, 0]) + float64(offset[r, 0]),
                float64(raw[r, 1]) + float64(offset[r, 1]),
                float64(target[r, 0]),
            )
            total += float64(weight[r]) * value
            mass += float64(weight[r])
        output[0] = total / mass


@cuda.jit
def normal_compare_rows(target, offset, before, after, output):
    r = cuda.grid(1)
    if r < before.shape[0]:
        m0, l0 = float64(before[r, 0]), float64(before[r, 1])
        m1, l1 = float64(after[r, 0]), float64(after[r, 1])
        y, mo, lo = float64(target[r, 0]), float64(offset[r, 0]), float64(offset[r, 1])
        # Retain the objective's actual float32 scale/gradient/Fisher domain.
        # Domain checks run for every row, independently of comparison support.
        old_loss, _, _, _ = normal_row(m0 + mo, l0 + lo, y)
        new_loss, _, _, _ = normal_row(m1 + mo, l1 + lo, y)
        lower, upper, code = float64(0), float64(0), 3
        if math.isfinite(old_loss) and math.isfinite(new_loss):
            lower, upper, code = _normal_change(m0, l0, m1, l1, y, mo, lo)
        output[r, 0], output[r, 1], output[r, 2] = lower, upper, code
        output[r, 3] = 1 if m0 == m1 and l0 == l1 else 0


@cuda.jit
def normal_compare_reduce(rows, weight, output):
    if cuda.grid(1) == 0:
        total, mass = (float64(0), float64(0)), (float64(0), float64(0))
        code, unchanged = 0, 1
        for r in range(rows.shape[0]):
            code = max(code, int(rows[r, 2]))
            if rows[r, 3] == 0:
                unchanged = 0
            w = float64(weight[r]), float64(weight[r])
            total = _compare_add(total, _compare_mul((rows[r, 0], rows[r, 1]), w))
            mass = _compare_add(mass, w)
        lower, upper = _compare_div(total, mass)
        if not math.isfinite(lower) or not math.isfinite(upper):
            code = max(code, 2)
        output[0], output[1], output[2], output[3] = lower, upper, code, unchanged


@cuda.jit
def diagonal_direction(gradient, metric, natural, damping, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        for k in range(output.shape[1]):
            denominator = float64(metric[r, k]) + float64(damping) if natural else float64(1)
            output[r, k] = (
                -float64(gradient[r, k]) / denominator
                if metric[r, k] > 0 and denominator > 0
                else math.nan
            )


@cuda.jit
def direction_fields(direction, channel, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        output[r, 0], output[r, 1] = -direction[r, channel], float32(1)


@cuda.jit
def matrix_add_raw(raw, delta, coefficient, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        for k in range(output.shape[1]):
            output[r, k] = raw[r, k] + libdevice.fmul_rn(coefficient, delta[r, k])


@cuda.jit
def mapped_add_raw(raw, scalar, mapping, coefficient, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        for k in range(output.shape[1]):
            delta = libdevice.fmul_rn(scalar[r, 0], mapping[k])
            output[r, k] = raw[r, k] + libdevice.fmul_rn(coefficient, delta)


@cuda.jit
def vector_mapped_add_raw(raw, prediction, mapping, coefficient, output):
    r = cuda.grid(1)
    if r < output.shape[0]:
        for k in range(output.shape[1]):
            delta = float32(0)
            for channel in range(prediction.shape[1]):
                product = libdevice.fmul_rn(prediction[r, channel], mapping[channel * output.shape[1] + k])
                delta = libdevice.fadd_rn(delta, product)
            output[r, k] = libdevice.fadd_rn(raw[r, k], libdevice.fmul_rn(coefficient, delta))
