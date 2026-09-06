"""Raw mixed-feature and full vector-tree oracle using exhaustive row routing."""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np

from .data import CategoryMap, NumericBinning
from .runs import data_identity
from .scalar import newton_leaf, node_score, nonnegative, training_weights
from .vector import _matrix


@dataclass(frozen=True)
class Transformer:
    names: tuple
    kinds: tuple
    encoders: tuple

    @classmethod
    def fit(cls, values, *, names, kinds, bins=254):
        names, kinds = tuple(names), tuple(kinds)
        x = np.asarray(values, dtype=object)
        if x.ndim != 2 or min(x.shape) == 0 or x.shape[1] != len(names) or len(names) != len(kinds):
            raise ValueError("training shape must match feature schema")
        if any(not isinstance(n, str) for n in names) or len(set(names)) != len(names):
            raise ValueError("feature names must be unique strings")
        if any(kind not in ("numeric", "categorical") for kind in kinds):
            raise ValueError("unknown feature kind")
        encoders = tuple(
            NumericBinning.fit(x[:, f], bins=bins)
            if kind == "numeric"
            else CategoryMap.fit(x[:, f])
            for f, kind in enumerate(kinds)
        )
        return cls(names, kinds, encoders)

    def identity(self, row_ids, values):
        self.transform(values)  # validate against fitted schema before binding
        metadata = {
            "version": "mixed-reference-v1",
            "kinds": self.kinds,
            "encoders": [
                {"cuts": encoder.cuts} if kind == "numeric" else {"categories": encoder.values}
                for kind, encoder in zip(self.kinds, self.encoders, strict=True)
            ],
        }
        return data_identity(row_ids, values, self.names, metadata)

    def transform(self, values):
        x = np.asarray(values, dtype=object)
        if x.ndim != 2 or len(x) == 0 or x.shape[1] != len(self.names):
            raise ValueError("prediction shape differs from fitted schema")
        result = np.empty(x.shape, dtype=float)
        for f, encoder in enumerate(self.encoders):
            codes, missing = encoder.transform(x[:, f])
            result[:, f] = [np.nan if m else c for c, m in zip(codes, missing, strict=True)]
        return result


def _route(x, rows, condition, kinds):
    feature, code, missing_left = condition
    left, right = [], []
    for row in rows:
        value = x[row, feature]
        selected = (
            missing_left
            if np.isnan(value)
            else (value == code if kinds[feature] == "categorical" else value <= code)
        )
        (left if selected else right).append(row)
    return tuple(left), tuple(right)


@dataclass(frozen=True)
class Node:
    rows: tuple
    depth: int
    value: tuple
    condition: tuple | None = None
    left: int = -1
    right: int = -1


@dataclass(frozen=True)
class MixedTree:
    transformer: Transformer
    nodes: tuple

    def predict(self, values):
        x = self.transformer.transform(values)
        output = []
        for row in range(len(x)):
            node = self.nodes[0]
            while node.condition is not None:
                left, _ = _route(x, (row,), node.condition, self.transformer.kinds)
                node = self.nodes[node.left if left else node.right]
            output.append(node.value)
        return np.array(output)


def grow(
    values,
    gradient,
    curvature,
    transformer,
    *,
    weight=None,
    projection=None,
    max_depth=2,
    policy="depthwise",
    reg_lambda=1.0,
):
    x = transformer.transform(values)
    g, h = _matrix(gradient, "gradient"), _matrix(curvature, "curvature")
    if g.shape != h.shape or len(g) != len(x) or np.any(h < 0):
        raise ValueError("aligned gradient and nonnegative curvature required")
    if not isinstance(max_depth, Integral) or isinstance(max_depth, bool) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    if policy not in ("depthwise", "best_first", "symmetric"):
        raise ValueError("unknown growth policy")
    reg_lambda = nonnegative(reg_lambda, "reg_lambda")
    w = training_weights(weight, len(x))
    p = np.eye(g.shape[1]) if projection is None else _matrix(projection, "projection")
    if p.shape[0] != g.shape[1] or np.any(np.sum(p * p, axis=0) == 0):
        raise ValueError("projection requires K rows and nonzero columns")
    sg, sh = (g @ p) * w[:, None], (h @ (p * p)) * w[:, None]
    lg, lh = g * w[:, None], h * w[:, None]
    if not all(np.all(np.isfinite(a)) for a in (sg, sh, lg, lh)):
        raise ValueError("non-finite statistics")
    conditions = tuple(
        (f, int(code), missing_left)
        for f in range(x.shape[1])
        for code in sorted(set(x[~np.isnan(x[:, f]), f]))
        for missing_left in (False, True)
    )

    def sums(fields, rows):
        return np.array([sum(fields[i, k] for i in rows) for k in range(fields.shape[1])])

    def score(rows):
        gs, hs = sums(sg, rows), sums(sh, rows)
        result = sum(node_score(a, b, reg_lambda=reg_lambda) for a, b in zip(gs, hs, strict=True))
        if not np.isfinite(result):
            raise ValueError("non-finite vector score")
        return result

    def leaf(rows, depth):
        gs, hs = sums(lg, rows), sums(lh, rows)
        return Node(
            rows,
            depth,
            tuple(newton_leaf(a, b, reg_lambda=reg_lambda) for a, b in zip(gs, hs, strict=True)),
        )

    def candidates(node):
        parent = score(node.rows)
        choices = {}
        for condition in conditions:
            left, right = _route(x, node.rows, condition, transformer.kinds)
            if not left or not right or np.any(sums(sh, left) <= 0) or np.any(sums(sh, right) <= 0):
                continue
            gain = score(left) + score(right) - parent
            if not np.isfinite(gain):
                raise ValueError("non-finite candidate gain")
            choices[condition] = (gain, left, right)
        return choices

    nodes = [leaf(tuple(range(len(x))), 0)]
    leaves = {0}
    while True:
        active = sorted(i for i in leaves if nodes[i].depth < max_depth)
        if not active:
            break
        by_node = {i: candidates(nodes[i]) for i in active}
        if policy == "symmetric":
            common = set.intersection(*(set(by_node[i]) for i in active))
            if not common:
                break
            totals = {c: sum(by_node[i][c][0] for i in active) for c in common}
            if not all(np.isfinite(value) for value in totals.values()):
                raise ValueError("non-finite symmetric layer gain")
            condition = min(common, key=lambda c: (-totals[c], c))
            if totals[condition] <= 0:
                break
            chosen = [(i, condition) for i in active]
        else:
            options = []
            for i in active:
                legal = [c for c in by_node[i] if by_node[i][c][0] > 0]
                if legal:
                    condition = min(legal, key=lambda c: (-by_node[i][c][0], c))
                    options.append((i, condition))
            if not options:
                break
            if policy == "best_first":
                chosen = [
                    min(options, key=lambda pair: (-by_node[pair[0]][pair[1]][0], pair[0], pair[1]))
                ]
            else:
                depth = min(nodes[i].depth for i, _ in options)
                chosen = [(i, c) for i, c in options if nodes[i].depth == depth]
        for i, condition in sorted(chosen):
            _, left, right = by_node[i][condition]
            first = len(nodes)
            nodes[i] = replace(nodes[i], condition=condition, left=first, right=first + 1)
            nodes.extend((leaf(left, nodes[i].depth + 1), leaf(right, nodes[i].depth + 1)))
            leaves.remove(i)
            leaves.update((first, first + 1))
    return MixedTree(transformer, tuple(nodes))
