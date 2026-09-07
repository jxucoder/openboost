"""Initial CUDA field/reduction kernels; invoked only on an owned context stream."""

import math

from numba import cuda, float32


@cuda.jit
def validate_fields(values, nonnegative, flags):
    q = cuda.grid(1)
    if q < values.shape[1]:
        invalid = 0
        for r in range(values.shape[0]):
            v = values[r, q]
            if not math.isfinite(v) or (nonnegative and v < 0):
                invalid = 1
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
