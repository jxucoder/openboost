"""Independent D1 geometry and exhaustive-tree traces for installed checks."""

import json
import sys
from pathlib import Path

import numpy as np
from tests.v1.reference.author import expectile, expectile_base
from tests.v1.reference.tree import fit_tree


def evidence():
    values = [[0], [1], [2], [3], [4], [None]]
    x = np.asarray(values, dtype=float)
    y = np.array([-3, 0, 2, 2, 90, 7.0])
    weight = np.array([2, 1, 3, 1, 0, 2.0])
    offset = np.array([1, -1, 0, 2, 1, -2.0])
    base = expectile_base(y - offset, weight=weight)
    raw = np.full(6, base)
    trace = []
    for _ in range(2):
        loss, g, h = expectile(raw + offset, y, weight=weight)
        tree = fit_tree(x, g, h, weight=weight, max_depth=2)
        raw = raw + 0.1 * tree.predict(x)
        trace.append(dict(loss=loss, gradient=g.tolist(), curvature=h.tolist(), raw=raw.tolist()))
    return dict(
        values=values,
        target=y.tolist(),
        weight=weight.tolist(),
        offset=offset.tolist(),
        base=base,
        trace=trace,
    )


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(json.dumps(evidence(), indent=2, allow_nan=False) + "\n")
