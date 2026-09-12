"""Independent stable row filtering and recursive leaf regions; no device imports."""

import numpy as np


def partition(codes, missing, rows, key):
    feature, threshold, missing_left = key
    left = {r for r in rows if (missing_left if missing[feature, r]
                               else codes[feature, r] <= threshold)}
    return ([r for r in rows if r in left], [r for r in rows if r not in left])


def predict(codes, missing, topology, values):
    """Assign whole leaf regions, independently of the per-row CUDA traversal."""
    values = np.asarray(values, np.float32).reshape(len(topology), -1)
    output = np.empty((np.asarray(codes).shape[1], values.shape[1]), np.float32)

    def visit(node, rows):
        feature, threshold, missing_left, left, right = topology[node]
        if feature == -1:
            output[rows] = values[node]
        else:
            children = partition(codes, missing, rows, (feature, threshold, missing_left))
            visit(left, children[0])
            visit(right, children[1])

    visit(0, list(range(len(output))))
    return output
