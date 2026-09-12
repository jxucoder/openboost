"""Exhaustive rational row-partition oracle; no production bins, heaps or gains."""

from fractions import Fraction as F
from itertools import product


def fit(
    x,
    fields,
    *,
    policy,
    depth=2,
    leaves=None,
    regularization=1,
    penalty=0,
    categorical=(),
    minimum=0,
    information_minima=None,
):
    fields = tuple(tuple(F(float(v)) for v in row) for row in fields)
    regularization, penalty = F(regularization), F(penalty)
    leaves = len(x) if leaves is None else leaves
    keys = tuple(
        (f, threshold, missing)
        for f in range(len(x[0]))
        for threshold, missing in product(
            sorted({r[f] for r in x if r[f] is not None}), (False, True)
        )
    )
    nodes = []

    def total(rows):
        return tuple(sum((fields[i][q] for i in rows), F(0)) for q in (0, 1))

    def score(rows):
        g, h = total(rows)
        return g * g / (2 * (h + regularization))

    def append(rows, level):
        g, h = total(rows)
        nodes.append(
            dict(
                rows=tuple(rows),
                level=level,
                key=None,
                left=-1,
                right=-1,
                value=-g / (h + regularization) if h + regularization else F(0),
            )
        )
        return len(nodes) - 1

    def candidates(i):
        rows = nodes[i]["rows"]
        result = {}
        for key in keys:
            f, t, missing = key
            parts = ([], [])
            for r in rows:
                v = x[r][f]
                left = missing if v is None else v == t if f in categorical else v <= t
                parts[0 if left else 1].append(r)
            if (
                all(parts)
                and min(total(side)[1] for side in parts) > 0
                and min(total(side)[1] for side in parts) >= F(minimum)
                and all(
                    sum((fields[i][q] for i in side), F(0)) >= F(limit)
                    for q, limit in (information_minima or {}).items()
                    for side in parts
                )
            ):
                result[key] = (score(parts[0]) + score(parts[1]) - score(rows) - penalty, parts)
        return result

    append(range(len(x)), 0)
    active, frontier = {0}, [0]
    while frontier and len(active) < leaves:
        if policy == "symmetric":
            if nodes[frontier[0]]["level"] >= depth or 2 * len(active) > leaves:
                break
            options = {i: candidates(i) for i in frontier}
            common = set.intersection(*(set(v) for v in options.values()))
            totals = {k: sum((options[i][k][0] for i in frontier), F(0)) for k in common}
            if not common:
                break
            key = min(common, key=lambda k: (-totals[k], k))
            chosen = [(i, key, *options[i][key]) for i in frontier] if totals[key] > 0 else []
        else:
            choices = []
            for i in sorted(active) if policy == "best_first" else frontier:
                if nodes[i]["level"] >= depth:
                    continue
                options = candidates(i)
                if options:
                    key = min(options, key=lambda k: (-options[k][0], k))
                    gain, parts = options[key]
                    if gain > 0:
                        choices.append((i, key, gain, parts))
            count = 1 if policy == "best_first" else leaves - len(active)
            chosen = sorted(choices, key=lambda c: (-c[2], c[0], c[1]))[:count]
        if not chosen:
            break
        frontier = []
        for i, key, _gain, parts in sorted(chosen):
            node = nodes[i]
            node["key"] = key
            node["left"] = append(parts[0], node["level"] + 1)
            node["right"] = append(parts[1], node["level"] + 1)
            active.remove(i)
            active.update((node["left"], node["right"]))
            frontier.extend((node["left"], node["right"]))
    return nodes


def predict(nodes, x, *, categorical=()):
    result = []
    for row in x:
        i = 0
        while nodes[i]["key"] is not None:
            f, t, missing = nodes[i]["key"]
            v = row[f]
            left = missing if v is None else v == t if f in categorical else v <= t
            i = nodes[i]["left" if left else "right"]
        result.append(float(nodes[i]["value"]))
    return result
