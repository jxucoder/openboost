"""111 independent joint K-tree trajectories and three comparison anchors."""

import numpy as np

from .device_multiclass import base, geometry
from .device_rounds import predict, tree
from .multiclass_comparison import compare

SETTINGS = (("fixed", 0.25), ("backtracking", 0.25), ("backtracking", 8.0), ("backtracking", 0.0))


def comparison(data, before, after):
    return compare(before, after, data["target"], data["offset"], data["weight"])


def fit(train, validation, *, depth, step, rate, count=3, patience=2, min_delta=0.01):
    def value(data, raw):
        return geometry(raw, data["target"], data["offset"], data["weight"])

    width = train["offset"].shape[1]
    initial = base(train["target"], width)
    raw = np.broadcast_to(initial, train["offset"].shape).copy()
    val = np.broadcast_to(initial, validation["offset"].shape).copy()
    best, anchor = val.copy(), val.copy()
    best_score = value(validation, val)[0]
    terms, best_terms, version, stale, history = 0, 0, 0, 0, []
    for _ in range(count):
        _, g, h, _, _ = value(train, raw)
        fields = [
            np.column_stack((g[:, j], h[:, j])) * np.asarray(train["weight"], np.float32)[:, None]
            for j in range(width)
        ]
        nodes = [tree(train["x"], field, depth) for field in fields]
        for channel in nodes:
            for node in channel:
                node["value"] = float(np.float32(node["value"]))
        predictions = [
            np.column_stack([predict(channel, data["x"]) for channel in nodes]).astype(np.float32)
            for data in (train, validation)
        ]
        trials = []
        for j in range(1 if step == "fixed" else 6):
            coefficient = float(np.float32(rate)) * 0.5**j
            candidate, candidate_val = [
                old + np.float32(coefficient) * prediction
                for old, prediction in zip((raw, val), predictions, strict=True)
            ]
            try:
                value(train, candidate)
                score = value(validation, candidate_val)[0]
                change = comparison(train, raw, candidate)
            except (ValueError, FloatingPointError, OverflowError):
                if step == "fixed":
                    raise
                trials.append(dict(coefficient=coefficient, accepted=False, failure=True))
                continue
            accepted = step == "fixed" or change.improves()
            trials.append(dict(coefficient=coefficient, accepted=accepted, failure=False))
            if accepted:
                raw, val = candidate, candidate_val
                terms, version = terms + width, version + 1
                if comparison(validation, best, val).improves():
                    best, best_score, best_terms = val.copy(), score, terms
                break
        change = comparison(validation, anchor, val)
        if change.improves(min_delta):
            anchor, stale = val.copy(), 0
        else:
            stale += 1
        history.append(dict(
            fields=fields, nodes=nodes, trials=trials, raw=raw.copy(), val=val.copy(),
            loss=value(train, raw)[0], score=value(validation, val)[0], best_score=best_score,
            terms=terms, version=version, best_terms=best_terms, stale=stale,
        ))
        if patience is not None and stale >= patience:
            break
    return dict(
        initial=initial, history=history, raw=raw, val=val, best=best, anchor=anchor,
        terms=terms, best_terms=best_terms, stale=stale,
        reason="patience" if patience is not None and stale >= patience else "budget",
    )
