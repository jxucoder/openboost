"""Exact original-row Newton mathematics; no NumPy, histogram or core imports."""

import struct
from fractions import Fraction as F

X = ((0, 3), (None, 2), (1, 1), (1, 0), (2, None), (2, 2), (None, 1), (3, 0))
GRADIENT = (0., .042141210703726076, .07697524756812128, .044312313692044324,
            0., 2.0064142014852084, -2.0121120875434078, .7221702433864421)
CURVATURE = (0., 1., 2., 1., 0., 3., 1., 1.)


def captured(*, stored=False):
    """IEEE input conversion only; all later arithmetic is exact rational math."""
    def number(value):
        if stored:
            value = struct.unpack("<f", struct.pack("<f", value))[0]
        return F(value)
    return X, [(number(g), number(h)) for g, h in zip(GRADIENT, CURVATURE, strict=True)]


def total(fields, rows):
    return tuple(sum((fields[r][q] for r in rows), F(0)) for q in (0, 1))


def node_score(sums):
    g, h = sums
    return g*g / (2*(h+1))


def candidates(x, fields, rows):
    """Global observed conditions; selected rows partitioned directly, no bins."""
    rows = tuple(rows)
    parent = total(fields, rows)
    result = []
    for feature in range(len(x[0])):
        thresholds = sorted({row[feature] for row in x if row[feature] is not None})
        for threshold in thresholds:
            for missing_left in (False, True):
                parts = ([], [])
                for r in rows:
                    value = x[r][feature]
                    left = missing_left if value is None else value <= threshold
                    parts[0 if left else 1].append(r)
                parts = tuple(tuple(side) for side in parts)
                sums = tuple(total(fields, side) for side in parts)
                legal = bool(all(parts) and min(side[1] for side in sums) > 0)
                gain = sum((node_score(side) for side in sums), F(0)) - node_score(parent) if legal else F(0)
                result.append(dict(key=(feature, threshold, missing_left), rows=parts, sums=sums,
                                   parent=parent, legal=legal, gain=gain))
    return result


def winner(options):
    valid = [r for r in options if r["legal"] and r["gain"] > 0]
    return min(valid, key=lambda r: (-r["gain"], r["key"])) if valid else None


def exact_tree(x, fields, *, depth=2):
    nodes = []

    def append(rows, level):
        g, h = total(fields, rows)
        nodes.append(dict(rows=tuple(rows), level=level, key=None, left=-1, right=-1, value=-g/(h+1)))
        return len(nodes)-1

    append(range(len(x)), 0)
    index = 0
    while index < len(nodes):
        node = nodes[index]
        selected = winner(candidates(x, fields, node["rows"])) if node["level"] < depth else None
        if selected:
            node["key"] = selected["key"]
            node["left"] = append(selected["rows"][0], node["level"]+1)
            node["right"] = append(selected["rows"][1], node["level"]+1)
        index += 1
    return nodes


def predict(nodes, x):
    result = []
    for row in x:
        i = 0
        while nodes[i]["key"] is not None:
            feature, threshold, missing_left = nodes[i]["key"]
            value = row[feature]
            left = missing_left if value is None else value <= threshold
            i = nodes[i]["left" if left else "right"]
        result.append(nodes[i]["value"])
    return result
