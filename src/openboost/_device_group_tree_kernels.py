"""M-wide routing and tree traversal; inputs are validated registered records."""

import math

from numba import cuda, float32


@cuda.jit
def partitions(codes, missing, rows, lengths, keys, active, output, sizes):
    m = cuda.grid(1)
    if m < rows.shape[0]:
        left, right = 0, 0
        if active[m]:
            f, t, missing_left = keys[m, 0], keys[m, 1], keys[m, 2] != 0
            for j in range(lengths[m]):
                r = rows[m, j]
                go_left = missing_left if missing[f, r] else codes[f, r] <= t
                if go_left:
                    output[m, 0, left] = r
                    left += 1
                else:
                    output[m, 1, right] = r
                    right += 1
        sizes[m, 0], sizes[m, 1] = left, right


@cuda.jit
def predictions(codes, missing, topology, values, active, output):
    i = cuda.grid(1)
    n = output.shape[1]
    if i < output.shape[0] * n:
        m, r = i // n, i % n
        node = 0
        if active[m]:
            while topology[m, node, 0] != -1:
                f, t = topology[m, node, 0], topology[m, node, 1]
                left = topology[m, node, 2] != 0 if missing[f, r] else codes[f, r] <= t
                node = topology[m, node, 3] if left else topology[m, node, 4]
        for q in range(output.shape[2]):
            output[m, r, q] = values[m, node, q] if active[m] else float32(0)


@cuda.jit
def finite_predictions(output, flags):
    m = cuda.grid(1)
    if m < output.shape[0]:
        invalid = 0
        for r in range(output.shape[1]):
            for q in range(output.shape[2]):
                if not math.isfinite(output[m, r, q]):
                    invalid = 1
        flags[m] = invalid
