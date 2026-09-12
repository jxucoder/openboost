"""Independent AFT scalar trajectories, original-row trees and three anchors."""

import numpy as np

from .aft_comparison import compare
from .device_aft import base, geometry
from .device_glm import stored
from .device_rounds import predict, tree

SETTINGS = (("fixed", .25), ("backtracking", .25), ("backtracking", 8.), ("backtracking", 0.))


def comparison(data, before, after, sigma):
    return compare(before, after, data["lower"], data["event"], data["offset"], data["weight"], sigma)


def fit(train, validation, *, sigma, depth, step, rate, count=3, patience=2, min_delta=.01):
    def value(data, raw):
        return geometry(raw, data["lower"], data["event"], data["offset"], data["weight"], sigma)

    initial = base(train["lower"], train["offset"], train["weight"])
    raw = np.full(len(train["lower"]), initial)
    val = np.full(len(validation["lower"]), initial)
    best, anchor = val.copy(), val.copy()
    best_score = value(validation, val)[0]
    terms, best_terms, stale, history = 0, 0, 0, []
    for _ in range(count):
        _, g, h = value(train, raw)
        fields = stored(np.column_stack((g, h))*stored(train["weight"])[:, None])
        nodes = tree(train["x"], fields, depth)
        trials = []
        for j in range(1 if step == "fixed" else 6):
            coefficient = float(np.float32(rate))*0.5**j
            candidate = stored(raw+stored(coefficient*stored(predict(nodes, train["x"]))))
            candidate_val = stored(val+stored(coefficient*stored(predict(nodes, validation["x"]))))
            try:
                value(train, candidate)
                score = value(validation, candidate_val)[0]
                change = comparison(train, raw, candidate, sigma)
            except (ValueError, FloatingPointError, OverflowError):
                if step == "fixed":
                    raise
                trials.append(dict(coefficient=coefficient, accepted=False, failure=True))
                continue
            accepted = step == "fixed" or change.improves()
            trials.append(dict(coefficient=coefficient, accepted=accepted, failure=False))
            if accepted:
                raw, val = candidate, candidate_val
                terms += 1
                if comparison(validation, best, val, sigma).improves():
                    best, best_score, best_terms = val.copy(), score, terms
                break
        change = comparison(validation, anchor, val, sigma)
        if change.improves(min_delta):
            anchor, stale = val.copy(), 0
        else:
            stale += 1
        history.append(dict(fields=fields, nodes=nodes, trials=trials, raw=raw.copy(), val=val.copy(),
                            loss=value(train, raw)[0], score=value(validation, val)[0], best_score=best_score,
                            terms=terms, best_terms=best_terms, stale=stale))
        if stale >= patience:
            break
    return dict(initial=initial, history=history, raw=raw, val=val, best=best, anchor=anchor,
                terms=terms, best_terms=best_terms, stale=stale, reason="patience" if stale >= patience else "budget")
