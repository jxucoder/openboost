"""107 independent scalar trajectories: original-row trees and convex comparisons."""

import numpy as np

from .device_glm import base, geometry, stored
from .device_rounds import predict, tree
from .glm_comparison import compare

SETTINGS = (("fixed", 0.25), ("backtracking", 0.25), ("backtracking", 8.0), ("backtracking", 0.0))


def comparison(family, data, before, after):
    return compare(
        family,
        before,
        after,
        data["target"],
        data["offset"],
        data["weight"],
        data.get("exposure", np.ones(len(before))),
    )


def fit(family, train, validation, *, depth, step, rate, count=3, patience=2, min_delta=0.01):
    def value(data, raw):
        return geometry(
            family, raw, data["target"], data["offset"], data["weight"], data.get("exposure")
        )

    initial = base(family, train["target"], train["offset"], train["weight"], train.get("exposure"))
    raw = np.full(len(train["target"]), initial)
    val = np.full(len(validation["target"]), initial)
    best, anchor = val.copy(), val.copy()
    best_score = value(validation, val)[0]
    terms, best_terms, stale, history = 0, 0, 0, []
    for _ in range(count):
        _, g, h = value(train, raw)
        fields = stored(np.column_stack((g, h)) * stored(train["weight"])[:, None])
        nodes = tree(train["x"], fields, depth)
        trials = []
        for j in range(1 if step == "fixed" else 6):
            coefficient = float(np.float32(rate)) * 0.5**j
            candidate = stored(raw + stored(coefficient * stored(predict(nodes, train["x"]))))
            candidate_val = stored(
                val + stored(coefficient * stored(predict(nodes, validation["x"])))
            )
            try:
                value(train, candidate)  # Validate every training row before the trial decision.
                score = value(validation, candidate_val)[0]
                change = comparison(family, train, raw, candidate)
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
                if comparison(family, validation, best, val).improves():
                    best, best_score, best_terms = val.copy(), score, terms
                break
        change = comparison(family, validation, anchor, val)
        if change.improves(min_delta):
            anchor, stale = val.copy(), 0
        else:
            stale += 1
        history.append(
            dict(
                fields=fields,
                nodes=nodes,
                trials=trials,
                raw=raw.copy(),
                val=val.copy(),
                loss=value(train, raw)[0],
                score=value(validation, val)[0],
                best_score=best_score,
                terms=terms,
                best_terms=best_terms,
                stale=stale,
            )
        )
        if stale >= patience:
            break
    return dict(
        initial=initial,
        history=history,
        raw=raw,
        val=val,
        best=best,
        anchor=anchor,
        terms=terms,
        best_terms=best_terms,
        stale=stale,
        reason="patience" if stale >= patience else "budget",
    )
