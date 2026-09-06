"""All-pairs query-local logistic oracle; lambda weights are frozen, not differentiated."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .scalar import finite_vector


def _input(scores, relevance, row_ids, k):
    s = finite_vector(scores, "scores")
    rel = finite_vector(relevance, "relevance", len(s))
    if np.any(rel < 0) or np.any(rel != np.floor(rel)):
        raise ValueError("relevance must contain nonnegative integers")
    if not isinstance(k, Integral) or isinstance(k, bool) or k < 1:
        raise ValueError("k must be a positive integer")
    ids = tuple(range(len(s))) if row_ids is None else tuple(row_ids)
    if (
        len(ids) != len(s)
        or any(not isinstance(i, Integral) or isinstance(i, bool) for i in ids)
        or len(set(ids)) != len(ids)
    ):
        raise ValueError("row_ids must be aligned unique integers")
    with np.errstate(over="ignore"):
        gains = np.exp2(rel) - 1
    if not np.all(np.isfinite(gains)):
        raise ValueError("relevance gains exceed float64")
    return s, rel, ids, gains


def _discount(rank, k):
    return 1 / np.log2(rank + 2) if rank < k else 0.0


def _ideal(gains, k):
    value = sum(g * _discount(i, k) for i, g in enumerate(sorted(gains, reverse=True)))
    if not np.isfinite(value):
        raise ValueError("ideal DCG exceeds float64")
    return value


def query_ndcg(scores, relevance, *, row_ids=None, k=10):
    s, _, ids, gains = _input(scores, relevance, row_ids, k)
    ideal = _ideal(gains, k)
    if ideal == 0:
        return 1.0
    order = sorted(range(len(s)), key=lambda i: (-s[i], ids[i]))
    return float(sum(gains[i] * _discount(rank, k) for rank, i in enumerate(order)) / ideal)


@dataclass(frozen=True)
class PairResult:
    loss: float
    gradient: np.ndarray
    curvature: np.ndarray
    pairs: tuple[tuple[int, int, float], ...]  # row positions and effective frozen weight


def pairwise(
    scores,
    relevance,
    query,
    *,
    row_ids=None,
    query_weight=None,
    pair_weight=None,
    weight=None,
    lambdas=False,
    k=10,
):
    if weight is not None:
        raise ValueError("ranking requires explicit query/pair weights, not row weight")
    s, rel, ids, gains = _input(scores, relevance, row_ids, k)
    groups = tuple(query)
    if len(groups) != len(s) or not groups:
        raise ValueError("query must align with nonempty scores")
    if any(not isinstance(q, (str, Integral)) or isinstance(q, bool) for q in groups):
        raise ValueError("query IDs must be strings or integers")
    grouped = {q: [i for i, g in enumerate(groups) if g == q] for q in dict.fromkeys(groups)}
    qw = {} if query_weight is None else dict(query_weight)
    pw = {} if pair_weight is None else dict(pair_weight)
    all_pairs = {(i, j) for rows in grouped.values() for i in rows for j in rows if rel[i] > rel[j]}
    if not set(qw) <= set(grouped) or not set(pw) <= all_pairs:
        raise ValueError(
            "weight keys must identify existing queries or eligible row-position pairs"
        )
    if any(not np.isscalar(w) or not np.isfinite(w) or w < 0 for w in (*qw.values(), *pw.values())):
        raise ValueError("weights must be finite and nonnegative")
    g, h = np.zeros(len(s)), np.zeros(len(s))
    loss, trace = 0.0, []
    for q, rows in grouped.items():
        pairs = [(i, j) for i in rows for j in rows if rel[i] > rel[j]]
        order = sorted(rows, key=lambda i: (-s[i], ids[i]))
        rank = {i: r for r, i in enumerate(order)}
        ideal = _ideal(gains[rows], k)
        for i, j in pairs:
            factor = qw.get(q, 1.0) * pw.get((i, j), 1.0) / len(pairs)
            if lambdas:
                delta = (
                    0.0
                    if ideal == 0
                    else abs(
                        (gains[i] - gains[j]) * (_discount(rank[i], k) - _discount(rank[j], k))
                    )
                    / ideal
                )
                factor *= delta
            difference = float(s[i]) - float(s[j])
            if not np.isfinite(difference):
                raise ValueError("score difference exceeds float64")
            tail = np.exp(-abs(difference))
            prob = tail / (1 + tail) if difference >= 0 else 1 / (1 + tail)
            curvature = tail / (1 + tail) ** 2
            loss += factor * np.logaddexp(0, -difference)
            g[i] -= factor * prob
            g[j] += factor * prob
            h[i] += factor * curvature
            h[j] += factor * curvature
            trace.append((i, j, float(factor)))
    if not np.isfinite(loss) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
        raise ValueError("non-finite pair reduction")
    return PairResult(float(loss), g, h, tuple(trace))
