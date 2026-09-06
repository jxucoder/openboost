"""Exhaustive fixed-bin scalar tree reference, intentionally without histograms.

Rows are enumerated for every candidate. This deliberately differs from the
planned histogram/prefix-sum production algorithm. Only tiny numeric fixtures
are supported; NaN is an explicit missing marker, never a regular bin code.
"""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from .scalar import (
    newton_leaf,
    node_score,
    nonnegative,
    row_ids,
    row_statistics,
    squared_error,
    sum_rows,
    weighted_mean,
)


def numeric_bins(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 2 or min(x.shape) == 0 or np.any(np.isinf(x)):
        raise ValueError("bins must be a nonempty numeric matrix with optional NaN")
    observed = x[~np.isnan(x)]
    if np.any(observed < 0) or np.any(observed != np.floor(observed)):
        raise ValueError("bins must contain nonnegative integer codes or NaN")
    return x


@dataclass(frozen=True, order=True)
class Condition:
    feature: int
    threshold: int
    missing_left: bool  # False sorts first: direction 0=right, 1=left.


@dataclass(frozen=True)
class Candidate:
    condition: Condition
    left: tuple[int, ...]
    right: tuple[int, ...]
    gain: float


def _conditions(x):
    # Include the last observed bin: it can separate observed rows from missing.
    return tuple(
        Condition(f, int(code), missing_left)
        for f in range(x.shape[1])
        for code in sorted(set(x[~np.isnan(x[:, f]), f]))
        for missing_left in (False, True)
    )


def _route(x, rows, condition):
    left, right = [], []
    for row in rows:
        value = x[row, condition.feature]
        go_left = condition.missing_left if np.isnan(value) else value <= condition.threshold
        (left if go_left else right).append(row)
    return tuple(left), tuple(right)


def enumerate_splits(
    bins,
    gradient,
    curvature,
    *,
    weight=None,
    rows=None,
    reg_lambda=1.0,
    split_penalty=0.0,
    min_child_h=0.0,
    information=None,
    min_information=0.0,
):
    """Return every feasible candidate, including those with nonpositive gain.

    Child H must be positive and meet min_child_h. Independent information
    fields are summed without training weights. Gain includes one split penalty.
    Keeping nonpositive candidates is essential for symmetric level decisions.
    """
    x = numeric_bins(bins)
    fields = row_statistics(gradient, curvature, weight)
    if len(fields) != len(x):
        raise ValueError("derivatives and bins must have the same rows")
    selected = row_ids(range(len(x)) if rows is None else rows, len(x))
    regularization = nonnegative(reg_lambda, "reg_lambda")
    penalty = nonnegative(split_penalty, "split_penalty")
    min_h = nonnegative(min_child_h, "min_child_h")
    min_info = nonnegative(min_information, "min_information")
    info = None if information is None else np.asarray(information, dtype=np.float64)
    if info is not None and (
        info.ndim != 2
        or info.shape[0] != len(x)
        or info.shape[1] == 0
        or not np.all(np.isfinite(info))
        or np.any(info < 0)
    ):
        raise ValueError("information must be a finite nonnegative [N,C] matrix")
    if info is None and min_info > 0:
        raise ValueError("min_information requires information fields")
    if not selected:
        return ()
    parent_g, parent_h = sum_rows(fields, selected)
    parent_score = node_score(parent_g, parent_h, reg_lambda=regularization)
    result = []
    for condition in _conditions(x):
        left, right = _route(x, selected, condition)
        if not left or not right:
            continue
        gl, hl = sum_rows(fields, left)
        gr, hr = sum_rows(fields, right)
        if hl <= 0 or hr <= 0 or min(hl, hr) < min_h:
            continue
        if info is not None and (
            np.any(sum_rows(info, left) < min_info) or np.any(sum_rows(info, right) < min_info)
        ):
            continue
        gain = (
            node_score(gl, hl, reg_lambda=regularization)
            + node_score(gr, hr, reg_lambda=regularization)
            - parent_score
            - penalty
        )
        if not np.isfinite(gain):
            raise ValueError("non-finite split gain")
        result.append(Candidate(condition, left, right, float(gain)))
    return tuple(result)


def best_split(candidates):
    eligible = [candidate for candidate in candidates if candidate.gain > 0]
    return min(eligible, key=lambda c: (-c.gain, c.condition)) if eligible else None


@dataclass(frozen=True)
class Node:
    rows: tuple[int, ...]
    depth: int
    value: float
    condition: Condition | None = None
    left: int = -1
    right: int = -1


@dataclass(frozen=True)
class Tree:
    nodes: tuple[Node, ...]
    n_features: int

    def predict(self, bins):
        x = numeric_bins(bins)
        if x.shape[1] != self.n_features:
            raise ValueError("bins feature count differs from the tree")
        result = []
        for row in range(len(x)):
            node = self.nodes[0]
            while node.condition is not None:
                left, _ = _route(x, (row,), node.condition)
                node = self.nodes[node.left if left else node.right]
            result.append(node.value)
        return np.array(result, dtype=np.float64)


def fit_tree(
    bins,
    gradient,
    curvature,
    *,
    weight=None,
    reg_lambda=1.0,
    split_penalty=0.0,
    min_child_h=0.0,
    information=None,
    min_information=0.0,
    policy="depthwise",
    max_depth=2,
    max_leaves=None,
):
    """Brute-force growth; best-first rescans leaves rather than using a heap."""
    x = numeric_bins(bins)
    if policy not in {"depthwise", "best_first", "symmetric"}:
        raise ValueError("unknown growth policy")
    if not isinstance(max_depth, Integral) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if max_leaves is None:
        max_leaves = len(x)
    if not isinstance(max_leaves, Integral) or isinstance(max_leaves, bool) or max_leaves < 1:
        raise ValueError("max_leaves must be a positive integer")
    kwargs = dict(
        weight=weight,
        reg_lambda=reg_lambda,
        split_penalty=split_penalty,
        min_child_h=min_child_h,
        information=information,
        min_information=min_information,
    )
    # Validate even when the caller requests a root-only tree.
    root_candidates = enumerate_splits(x, gradient, curvature, **kwargs)
    fields = row_statistics(gradient, curvature, weight)

    def leaf(rows, depth):
        g, h = sum_rows(fields, rows)
        return Node(rows, depth, newton_leaf(g, h, reg_lambda=reg_lambda))

    nodes = [leaf(tuple(range(len(x))), 0)]

    def candidates(node_id):
        if node_id == 0:
            return root_candidates
        return enumerate_splits(x, gradient, curvature, rows=nodes[node_id].rows, **kwargs)

    def split(node_id, candidate):
        node = nodes[node_id]
        left, right = len(nodes), len(nodes) + 1
        nodes[node_id] = replace(node, condition=candidate.condition, left=left, right=right)
        nodes.extend((leaf(candidate.left, node.depth + 1), leaf(candidate.right, node.depth + 1)))
        return left, right

    leaves, frontier = {0}, [0]
    while frontier and len(leaves) < max_leaves:
        if policy == "best_first":
            choices = [
                (i, best_split(candidates(i))) for i in sorted(leaves) if nodes[i].depth < max_depth
            ]
            choices = [(i, candidate) for i, candidate in choices if candidate is not None]
            if not choices:
                break
            chosen = [min(choices, key=lambda item: (-item[1].gain, item[0], item[1].condition))]
        elif policy == "symmetric":
            if nodes[frontier[0]].depth >= max_depth or 2 * len(leaves) > max_leaves:
                break
            by_node = {i: {c.condition: c for c in candidates(i)} for i in frontier}
            common = set.intersection(*(set(by_node[i]) for i in frontier))
            if not common:
                break
            totals = {
                condition: sum(by_node[i][condition].gain for i in frontier) for condition in common
            }
            condition = min(common, key=lambda c: (-totals[c], c))
            if totals[condition] <= 0:
                break
            chosen = [(i, by_node[i][condition]) for i in frontier]
        else:
            choices = [
                (i, best_split(candidates(i))) for i in frontier if nodes[i].depth < max_depth
            ]
            choices = [(i, candidate) for i, candidate in choices if candidate is not None]
            chosen = sorted(choices, key=lambda item: (-item[1].gain, item[0], item[1].condition))[
                : max_leaves - len(leaves)
            ]
        if not chosen:
            break
        next_frontier = []
        # Stable node IDs are independent of the ranking used to select a layer.
        for node_id, candidate in sorted(chosen):
            children = split(node_id, candidate)
            leaves.remove(node_id)
            leaves.update(children)
            next_frontier.extend(children)
        frontier = next_frontier
    return Tree(tuple(nodes), x.shape[1])


@dataclass(frozen=True)
class Step:
    gradient: tuple[float, ...]
    raw_before: tuple[float, ...]
    raw_after: tuple[float, ...]
    loss_before: float
    loss_after: float
    tree: Tree
    coefficient: float


@dataclass(frozen=True)
class BoostTrace:
    base: float
    n_features: int
    steps: tuple[Step, ...]

    def predict(self, bins):
        x = numeric_bins(bins)
        if x.shape[1] != self.n_features:
            raise ValueError("bins feature count differs from the model")
        raw = np.full(len(x), self.base, dtype=np.float64)
        for step in self.steps:
            raw += step.coefficient * step.tree.predict(x)
        return raw


def boost_squared(bins, y, *, weight=None, rounds=2, learning_rate=0.1, **tree_options):
    """A tiny fixed-step trace, not a trainer API or persistence implementation."""
    x = numeric_bins(bins)
    if not isinstance(rounds, Integral) or isinstance(rounds, bool) or rounds < 0:
        raise ValueError("rounds must be a nonnegative integer")
    coefficient = nonnegative(learning_rate, "learning_rate")
    base = weighted_mean(y, weight)
    raw = np.full(len(x), base, dtype=np.float64)
    # Validate aligned labels even if rounds=0.
    squared_error(raw, y, weight)
    steps = []
    for _ in range(rounds):
        loss_before, gradient, curvature = squared_error(raw, y, weight)
        tree = fit_tree(x, gradient, curvature, weight=weight, **tree_options)
        updated = raw + coefficient * tree.predict(x)
        loss_after, _, _ = squared_error(updated, y, weight)
        steps.append(
            Step(
                tuple(gradient),
                tuple(raw),
                tuple(updated),
                loss_before,
                loss_after,
                tree,
                coefficient,
            )
        )
        raw = updated
    return BoostTrace(base, x.shape[1], tuple(steps))
