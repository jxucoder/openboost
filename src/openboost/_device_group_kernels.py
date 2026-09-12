"""Grouped CUDA reductions; each cell follows its own original-row order."""

import math

from numba import cuda, float32


@cuda.jit
def histograms(codes, missing, bins, values, rows, lengths, active, sums, counts):
    i = cuda.grid(1)
    features, slots, width = sums.shape[1], sums.shape[2], sums.shape[3]
    if i < sums.size:
        m = i // (features * slots * width)
        f, b, q = (i // (slots * width)) % features, (i // width) % slots, i % width
        total, count = float32(0), 0
        if active[m] and b <= bins[f]:
            for j in range(lengths[m]):
                r = rows[m, j]
                actual = bins[f] if missing[f, r] else codes[f, r]
                if actual == b:
                    total = float32(total + values[m, r, q])
                    count += 1
        sums[m, f, b, q] = total
        if q == 0:
            counts[m, f, b] = count


@cuda.jit
def row_totals(values, rows, lengths, active, total):
    i = cuda.grid(1)
    width = total.shape[1]
    if i < total.size:
        m, q = i // width, i % width
        value = float32(0)
        if active[m]:
            for j in range(lengths[m]):
                value = float32(value + values[m, rows[m, j], q])
        total[m, q] = value


@cuda.jit
def finite_histograms(sums, total, flags):
    m = cuda.grid(1)
    if m < total.shape[0]:
        invalid = 0
        for q in range(total.shape[1]):
            if not math.isfinite(total[m, q]):
                invalid = 1
        for f in range(sums.shape[1]):
            for b in range(sums.shape[2]):
                for q in range(sums.shape[3]):
                    if not math.isfinite(sums[m, f, b, q]):
                        invalid = 1
        flags[m] = invalid
