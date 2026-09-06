"""D1/D3/D4 independent author-task oracles, not public extensions or an E5 result."""

from dataclasses import replace

import numpy as np

from .coupled import normal, step
from .positive import _result
from .quantile import _level, pinball
from .scalar import finite_vector, nonnegative, training_weights
from .tree import fit_tree


def expectile(raw, target, *, tau=0.8, weight=None):
    tau = _level(tau)
    y = finite_vector(target, "target")
    raw = finite_vector(raw, "raw", len(y))
    residual = y - raw
    asymmetry = np.where(residual < 0, 1 - tau, tau)
    # At r=0 the gradient is zero; h=2*tau is a declared branch convention.
    return _result(asymmetry * residual**2, -2 * asymmetry * residual, 2 * asymmetry, weight)


def expectile_base(target, *, tau=0.8, weight=None):
    tau = _level(tau)
    y = finite_vector(target, "target")
    w = training_weights(weight, len(y))
    y, w = y[w > 0], w[w > 0]
    boundaries = sorted(set(y))
    candidates = set(boundaries)
    # Each open interval fixes the asymmetric weights. Enumerate its stationary
    # point directly; no iteration using the objective gradient is needed.
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        mass = w * np.where(y <= low, 1 - tau, tau)
        value = sum((mass / sum(mass)) * y)
        if low <= value <= high:
            candidates.add(float(value))

    def loss(value):
        residual = y - value
        return sum(w * np.where(residual < 0, 1 - tau, tau) * residual**2)

    scored = [(loss(value), value) for value in candidates]
    if not all(np.isfinite(value) for value, _ in scored):
        raise ValueError("expectile base objective exceeds float64")
    return float(min(scored)[1])


def penalized_quantile(residual, q, weight=None, *, penalty, anchor):
    q = _level(q)
    residual = finite_vector(residual, "residual")
    w = training_weights(weight, len(residual))
    penalty = nonnegative(penalty, "penalty")
    if penalty == 0 or not np.isscalar(anchor) or not np.isfinite(anchor):
        raise ValueError("penalty must be positive and anchor finite")
    residual, w = residual[w > 0], w[w > 0]
    breaks = sorted(set(residual))
    candidates = set(breaks)
    boundaries = [-np.inf, *breaks, np.inf]
    for low, high in zip(boundaries[:-1], boundaries[1:], strict=True):
        left_mass = sum(w[residual <= low])
        value = anchor + (q * sum(w) - left_mass) / penalty
        if low <= value <= high and np.isfinite(value):
            candidates.add(float(value))

    def loss(value):
        r = residual - value
        return sum(w * np.maximum(q * r, (q - 1) * r)) + penalty * (value - anchor) ** 2 / 2

    values = [(loss(value), value) for value in candidates]
    if not all(np.isfinite(loss) for loss, _ in values):
        raise ValueError("penalized objective exceeds float64")
    return float(min(values)[1])


def penalized_tree(bins, raw, target, q, *, weight=None, penalty, anchor, **tree_options):
    _, g, h = pinball(raw, target, q, weight)
    residual = np.asarray(target, dtype=float) - np.asarray(raw, dtype=float)
    w = training_weights(weight, len(residual))
    # Validate solver config even if no split is chosen.
    penalized_quantile(residual, q, w, penalty=penalty, anchor=anchor)
    tree = fit_tree(bins, g, h, weight=weight, **tree_options)
    nodes = []
    for node in tree.nodes:
        if node.condition is None:
            rows = list(node.rows)
            node = replace(
                node,
                value=penalized_quantile(
                    residual[rows], q, w[rows], penalty=penalty, anchor=anchor
                ),
            )
        nodes.append(node)
    return replace(tree, nodes=tuple(nodes))


def ordered_normal(bins, raw, target, *, weight=None, objective=normal):
    """D4 exact ordered rule. Rejection passes the unchanged snapshot forward."""
    rates = tuple(0.1 * 0.5**j for j in range(6))
    results = []
    for channel in (0, 1):
        result = step(
            bins,
            raw,
            target,
            objective,
            weight=weight,
            mode="full",
            channels=(channel,),
            rates=rates,
            require_decrease=True,
        )
        results.append(result)
        raw = result.raw_after
    return tuple(results)
