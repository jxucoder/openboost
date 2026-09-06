"""Train-only dense encoding and output scaling for baseline evaluation adapters."""

import numpy as np


def fit_encoder(numeric, categories=None):
    x = np.asarray(numeric, dtype=float)
    if x.ndim != 2 or not len(x) or np.isinf(x).any():
        raise ValueError("numeric matrix with finite values or NaN required")
    median = []
    for column in x.T:
        observed = column[~np.isnan(column)]
        median.append(float(np.median(observed)) if len(observed) else 0.0)
    vocab = {}
    for name, values in sorted((categories or {}).items()):
        a = np.asarray(values, dtype=object)
        if a.shape != (len(x),) or any(v is not None and not isinstance(v, str) for v in a):
            raise ValueError("aligned string/None categories required")
        vocab[name] = sorted({v for v in a if v is not None})
    return {
        "numeric_columns": x.shape[1],
        "median": median,
        "categories": vocab,
        "unknown": "missing-indicator",
        "numeric_missing_indicators": True,
    }


def transform(encoder, numeric, categories=None):
    x = np.asarray(numeric, dtype=float)
    if x.ndim != 2 or x.shape[1] != encoder["numeric_columns"] or np.isinf(x).any():
        raise ValueError("wrong numeric schema")
    categories = categories or {}
    if set(categories) != set(encoder["categories"]):
        raise ValueError("category fields differ from fitted schema")
    parts = [np.where(np.isnan(x), encoder["median"], x), np.isnan(x).astype(float)]
    for name, vocab in encoder["categories"].items():
        a = np.asarray(categories[name], dtype=object)
        if a.shape != (len(x),) or any(v is not None and not isinstance(v, str) for v in a):
            raise ValueError("invalid category rows")
        lookup = {v: i for i, v in enumerate(vocab)}
        codes = np.array([lookup.get(v, len(vocab)) for v in a])
        encoded = np.zeros((len(x), len(vocab) + 1))
        encoded[np.arange(len(x)), codes] = 1
        parts.append(encoded)
    return np.concatenate(parts, axis=1)


def fit_target_scale(y):
    a = np.asarray(y, dtype=float)
    if a.ndim != 2 or not len(a) or not np.isfinite(a).all():
        raise ValueError("finite matrix targets required")
    std = a.std(axis=0)
    return {
        "mean": a.mean(axis=0).tolist(),
        "std": np.where(std == 0, 1.0, std).tolist(),
        "constant": (std == 0).tolist(),
    }


def censoring_support(time, event):
    """Training-only reverse Kaplan-Meier with event-before-censor tie handling.

    At tied times, remove observed events from the censoring risk denominator.
    Freeze the step function, its positivity support, and a training-quantile grid.
    Actual test evaluation must reject points outside this support.
    """
    t = np.asarray(time, dtype=float)
    e = np.asarray(event)
    if (
        t.ndim != 1
        or e.shape != t.shape
        or not len(t)
        or not np.isfinite(t).all()
        or np.any(t <= 0)
        or not np.isin(e, [0, 1]).all()
    ):
        raise ValueError("invalid observed times/events")
    survival = 1.0
    values = []
    times = []
    for point in np.unique(t):
        at = t == point
        risk = int((t >= point).sum()) - int((at & (e == 1)).sum())
        censored = int((at & (e == 0)).sum())
        if censored:
            survival *= 1 - censored / risk
        times.append(float(point))
        values.append(survival)
    stop = next((times[i] for i, v in enumerate(values) if v == 0), float(t.max()))
    grid = np.unique(np.quantile(t, np.linspace(0.1, 0.9, 9)))
    grid = grid[(grid < stop) & (grid < float(t.max()))]
    if not len(grid):
        raise ValueError("no supported censoring evaluation grid")
    return {
        "times": times,
        "survival": values,
        "strict_upper": stop,
        "grid": grid.tolist(),
        "tie_rule": "events removed before censoring risk",
    }
