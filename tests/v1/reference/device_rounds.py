"""Frozen 088 original-row tree and squared-round oracle; no production imports."""

import numpy as np

from .device_splits import enumerate_candidates, winner
from .device_splits import fixture as split_fixture


def fixture(case):
    source = "missing_weighted" if case == "weighted" else "d2"
    x, gradient, _, weight, information, _ = split_fixture(source)
    offset = (np.arange(len(x)) % 3 - 1) / 4
    target = -gradient + offset
    validation_x = x[::-1].copy()
    validation_offset = offset[::-1] / 2
    validation_target = -gradient[::-1] + validation_offset
    if case == "conflict":
        validation_target = gradient[::-1] + validation_offset
    return dict(
        x=x,
        target=target,
        offset=offset,
        weight=weight,
        information=information,
        validation_x=validation_x,
        validation_target=validation_target,
        validation_offset=validation_offset,
        validation_weight=np.ones(len(x)),
    )


def predict(nodes, x):
    result = np.empty(len(x))
    for r in range(len(x)):
        i = 0
        while nodes[i]["key"] is not None:
            f, threshold, missing_left = nodes[i]["key"]
            left = missing_left if np.isnan(x[r, f]) else x[r, f] <= threshold
            i = nodes[i]["left" if left else "right"]
        result[r] = nodes[i]["value"]
    return result


def tree(x, fields, depth, minimum=None):
    nodes = []

    def append(rows, level):
        total = np.zeros(fields.shape[1])
        for r in rows:
            total += fields[r]
        nodes.append(
            dict(
                key=None,
                left=-1,
                right=-1,
                value=-total[0] / (total[1] + 1),
                rows=list(rows),
                level=level,
            )
        )
        return len(nodes) - 1

    append(range(len(x)), 0)
    index = 0
    while index < len(nodes):
        node = nodes[index]
        if node["level"] < depth:
            options, _ = enumerate_candidates(x, fields, node["rows"], minimum=minimum or 0)
            chosen = winner(options, constrained=minimum is not None)
            if chosen:
                node["key"] = chosen["key"]
                node["left"] = append(chosen["rows"][0], node["level"] + 1)
                node["right"] = append(chosen["rows"][1], node["level"] + 1)
        index += 1
    return nodes


def loss(target, offset, weight, raw):
    total = 0.0
    for r in range(len(raw)):
        total += weight[r] * (raw[r] + offset[r] - target[r]) ** 2 / 2
    return total / sum(weight)


def rounds(case, *, depth=2, minimum=None, count=2, rate=0.5):
    f = fixture(case)
    base = sum(f["weight"] * (f["target"] - f["offset"])) / sum(f["weight"])
    raw = np.full(len(f["target"]), base)
    validation_raw = np.full(len(f["validation_target"]), base)
    def score(value):
        return loss(
            f["validation_target"], f["validation_offset"], f["validation_weight"], value
        )
    best_score, best_round = score(validation_raw), 0
    steps = []
    for i in range(count):
        gradient = raw + f["offset"] - f["target"]
        fields = np.column_stack((gradient * f["weight"], f["weight"], f["information"]))
        nodes = tree(f["x"], fields, depth, minimum)
        before = raw.copy()
        raw = raw + rate * predict(nodes, f["x"])
        validation_raw = validation_raw + rate * predict(nodes, f["validation_x"])
        validation_score = score(validation_raw)
        if validation_score < best_score:
            best_score, best_round = validation_score, i + 1
        steps.append(
            dict(
                gradient=gradient,
                fields=fields,
                nodes=nodes,
                before=before,
                raw=raw.copy(),
                validation_raw=validation_raw.copy(),
                validation_score=validation_score,
                loss=loss(f["target"], f["offset"], f["weight"], raw),
                best_score=best_score,
                best_round=best_round,
            )
        )
    return base, steps
