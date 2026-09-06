"""Pinball geometry and routed residual quantile leaves, independent of production."""

from dataclasses import replace

import numpy as np

from .scalar import finite_vector, training_weights
from .tree import fit_tree


def _level(q):
    if not np.isscalar(q) or not np.isfinite(q) or not 0 < q < 1:
        raise ValueError("q must be finite and strictly between zero and one")
    return float(q)


def weighted_quantile(values, q, weight=None):
    q = _level(q)
    values = finite_vector(values, "values")
    w = training_weights(weight, len(values))
    ordered = sorted((float(v), float(m)) for v, m in zip(values, w, strict=True) if m > 0)
    threshold = q * sum(m for _, m in ordered)
    cumulative = 0.0
    for value, mass in ordered:
        cumulative += mass
        if cumulative >= threshold:
            return value
    return ordered[-1][0]  # floating summation endpoint


def pinball(raw, target, q, weight=None):
    q = _level(q)
    y = finite_vector(target, "target")
    raw = finite_vector(raw, "raw", len(y))
    w = training_weights(weight, len(y))
    residual = y - raw
    if not np.all(np.isfinite(residual)):
        raise ValueError("non-finite residual")
    loss = np.maximum(q * residual, (q - 1) * residual)
    return float(sum((w / sum(w)) * loss)), (y < raw).astype(float) - q, np.ones(len(y))


def fit_quantile_tree(bins, raw, target, q, *, weight=None, **tree_options):
    _, g, pseudo_h = pinball(raw, target, q, weight)
    tree = fit_tree(bins, g, pseudo_h, weight=weight, **tree_options)
    residual = np.asarray(target, dtype=float) - np.asarray(raw, dtype=float)
    w = training_weights(weight, len(residual))
    nodes = []
    for node in tree.nodes:
        if node.condition is None:
            rows = list(node.rows)
            node = replace(node, value=weighted_quantile(residual[rows], q, w[rows]))
        nodes.append(node)
    return replace(tree, nodes=tuple(nodes))
