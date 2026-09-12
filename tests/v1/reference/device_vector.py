"""117 exact original-row diagonal Newton oracle; no production imports."""

from fractions import Fraction

import numpy as np


def fixture(width=2, projected=False, case="weighted"):
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1], [np.nan, 0], [2, np.nan]], float)
    first = np.array([-4, -4, 1, 1, 3, 3, -4, 3.0])
    second = np.array([-10, 10, -10, 10, -10, 10, -10, 10.0])
    g = np.column_stack([first, second, first / 2, second / 4])[:, :width]
    h = np.tile([1, 2, 0.5, 1.5], (len(x), 1))[:, :width]
    weight = np.array([1, 2, 1, 0, 2, 1, 1, 0.5])
    if case == "ties":
        x = np.tile([[0, 0], [0, 0], [1, 1], [1, 1]], (2, 1)).astype(float)
        g = np.tile([[-2], [-2], [2], [2]], (2, width)).astype(float)
        h = np.ones_like(g)
        weight = np.ones(len(x))
    elif case == "zero_channel":
        h[:, -1] = 0
    elif case != "weighted":
        raise ValueError(case)
    # A coordinate projection is exact at these stored inputs and has S=1.
    split_g, split_h = (g[:, :1], h[:, :1]) if projected else (g, h)
    return dict(x=x, weight=weight, g=g, h=h, split_g=split_g, split_h=split_h)


def total(values, rows):
    return tuple(
        sum((Fraction(float(values[r, k])) for r in rows), Fraction())
        for k in range(values.shape[1])
    )


def leaf(g, h, regularization=1):
    regularization = Fraction(regularization)
    if any(v < 0 or v + regularization <= 0 for v in h):
        raise ValueError("nonnegative curvature and positive denominator required")
    return tuple(-a / (b + regularization) for a, b in zip(g, h, strict=True))


def candidates(x, g, h, rows, *, regularization=1, penalty=0, minimum=0):
    """Exact rational sums/scores; enumerate conditions from original input rows."""
    regularization, penalty, minimum = map(Fraction, (regularization, penalty, minimum))
    parent_g, parent_h = total(g, rows), total(h, rows)

    def objective(gs, hs):
        return sum(
            (a * a / (2 * (b + regularization)) for a, b in zip(gs, hs, strict=True)), Fraction()
        )

    result = []
    for f in range(x.shape[1]):
        for threshold in sorted(set(x[np.isfinite(x[:, f]), f])):
            for missing_left in (False, True):
                sides = ([], [])
                for r in rows:
                    left = missing_left if np.isnan(x[r, f]) else x[r, f] <= threshold
                    sides[0 if left else 1].append(int(r))
                gs = tuple(total(g, side) for side in sides)
                hs = tuple(total(h, side) for side in sides)
                positive = all(sides) and all(v > 0 for side in hs for v in side)
                legal = positive and all(v >= minimum for side in hs for v in side)
                gain = (
                    (
                        objective(gs[0], hs[0])
                        + objective(gs[1], hs[1])
                        - objective(parent_g, parent_h)
                        - penalty
                    )
                    if positive
                    else Fraction()
                )
                result.append(
                    dict(
                        key=(f, int(threshold), missing_left),
                        rows=sides,
                        g=gs,
                        h=hs,
                        gain=gain,
                        legal=bool(legal),
                    )
                )
    return result


def winner(options):
    allowed = [c for c in options if c["legal"] and c["gain"] > 0]
    return min(allowed, key=lambda c: (-c["gain"], c["key"])) if allowed else None


def tree(source, depth, *, regularization=1, penalty=0, minimum=0):
    w = source["weight"][:, None]
    g, h = source["g"] * w, source["h"] * w
    split_g, split_h = source["split_g"] * w, source["split_h"] * w
    nodes, queue = [], [(list(range(len(g))), 0)]
    i = 0
    while i < len(queue):
        rows, level = queue[i]
        node = dict(
            key=None,
            left=-1,
            right=-1,
            rows=rows,
            value=leaf(total(g, rows), total(h, rows), regularization),
        )
        if level < depth:
            chosen = winner(
                candidates(
                    source["x"],
                    split_g,
                    split_h,
                    rows,
                    regularization=regularization,
                    penalty=penalty,
                    minimum=minimum,
                )
            )
            if chosen:
                node.update(key=chosen["key"], left=len(queue), right=len(queue) + 1)
                queue.extend((side, level + 1) for side in chosen["rows"])
        nodes.append(node)
        i += 1
    return nodes


def predict(nodes, x):
    output = []
    for row in x:
        i = 0
        while nodes[i]["key"] is not None:
            f, threshold, missing_left = nodes[i]["key"]
            left = missing_left if np.isnan(row[f]) else row[f] <= threshold
            i = nodes[i]["left" if left else "right"]
        output.append([float(v) for v in nodes[i]["value"]])
    return np.asarray(output)
