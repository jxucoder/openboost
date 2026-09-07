"""Composable CPU scalar/vector tree growth with validated mixed-feature inference state."""

import heapq
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ops
from .binning import BinnedData, Binning, _array
from .data import MixedData, NumericData, _identity, _owned
from .leaves import ResidualContext


def _indices(value):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < -1) or np.any(a > 2**31 - 1):
        raise ValueError("int32 topology vector required")
    return _array(a, "<i4")


@dataclass(frozen=True, eq=False)
class Tree:
    """Owned explicit topology; leaf feature/threshold/children use -1 sentinels.

    Predictions are learner outputs [N, L], without a base or offset.
    Training row membership is deliberately absent from inference artifacts.
    """

    binning: Binning
    feature: np.ndarray
    threshold: np.ndarray
    missing_left: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray

    def __post_init__(self):
        if not isinstance(self.binning, Binning):
            raise ValueError("feature transformer required")
        for name in ("feature", "threshold", "left", "right"):
            object.__setattr__(self, name, _indices(getattr(self, name)))
        value = np.asarray(self.value, dtype=float)
        values = _owned(value[:, None] if value.ndim == 1 else value, ndim=2)
        missing = np.asarray(self.missing_left)
        if missing.ndim != 1 or missing.dtype.kind != "b":
            raise ValueError("boolean missing routes required")
        object.__setattr__(self, "value", values)
        object.__setattr__(self, "missing_left", _array(missing, bool))
        n = len(values)
        if n > 2**31 - 1 or any(
            len(getattr(self, name)) != n
            for name in ("feature", "threshold", "missing_left", "left", "right")
        ):
            raise ValueError("aligned bounded topology required")
        seen, pending = set(), [0]
        while pending:
            i = pending.pop()
            if i < 0 or i >= n or i in seen:
                raise ValueError("tree has invalid children, a cycle or shared nodes")
            seen.add(i)
            f, t, left, right = (
                int(getattr(self, name)[i]) for name in ("feature", "threshold", "left", "right")
            )
            if f == -1:
                if (t, left, right) != (-1, -1, -1) or self.missing_left[i]:
                    raise ValueError("invalid leaf topology")
            else:
                if (
                    f >= len(self.binning.cuts)
                    or t < 0
                    or t
                    >= (
                        self.binning.bin_counts[f]
                        if self.binning.categories[f] is None
                        else len(self.binning.categories[f])
                    )
                ):
                    raise ValueError("split exceeds transformer schema")
                pending.extend((right, left))
        if len(seen) != n:
            raise ValueError("unreachable tree nodes")

    @property
    def identity(self):
        return _identity(self.record())

    @property
    def output_width(self):
        return self.value.shape[1]

    def predict(self, data, *, binned=None):
        if binned is None:
            binned = self.binning.transform(data)
        elif (
            not isinstance(data, (NumericData, MixedData))
            or not isinstance(binned, BinnedData)
            or binned.data.identity != data.identity
            or binned.binning.identity != self.binning.identity
        ):
            raise ValueError("prediction encoding data/binning identity differs")
        result = np.empty((len(data.values), self.output_width))
        pending = [(0, np.arange(len(data.values)))]
        while pending:
            i, rows = pending.pop()
            f = self.feature[i]
            if f == -1:
                result[rows] = self.value[i]
            else:
                mask = np.where(
                    binned.missing[f, rows],
                    self.missing_left[i],
                    binned.codes[f, rows] == self.threshold[i]
                    if self.binning.categories[f] is not None
                    else binned.codes[f, rows] <= self.threshold[i],
                )
                pending.extend(((self.left[i], rows[mask]), (self.right[i], rows[~mask])))
        return result

    def record(self):
        return dict(
            format="openboost-tree-v3",
            feature_names=list(self.binning.feature_names),
            cuts=[c.tolist() for c in self.binning.cuts],
            categories=[None if c is None else list(c) for c in self.binning.categories],
            **{
                name: getattr(self, name).tolist()
                for name in ("feature", "threshold", "missing_left", "left", "right", "value")
            },
        )

    def save(self, path):
        Path(path).write_text(json.dumps(self.record(), allow_nan=False) + "\n")

    @classmethod
    def load(cls, path):
        def pairs(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate artifact field")
                result[key] = value
            return result

        record = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
        return cls.from_record(record)

    @classmethod
    def from_record(cls, record):
        """Validate a nested inference record without a filesystem round trip."""
        vectors = ("feature", "threshold", "missing_left", "left", "right", "value")
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "feature_names", "cuts", "categories", *vectors}
            or record["format"] != "openboost-tree-v3"
            or any(
                not isinstance(record[name], list)
                for name in ("feature_names", "cuts", "categories", *vectors)
            )
            or any(not isinstance(c, list) for c in record["cuts"])
            or any(c is not None and not isinstance(c, list) for c in record["categories"])
            or any(not isinstance(row, list) for row in record["value"])
        ):
            raise ValueError("unsupported or corrupt tree artifact schema")
        return cls(
            Binning(record["feature_names"], record["cuts"], record["categories"]),
            **{name: record[name] for name in vectors},
        )


class _Growth:
    """Local construction scratch shared by ordinary policy loops."""

    def __init__(
        self,
        data,
        fields,
        max_depth,
        max_leaves,
        scoring,
        legality,
        leaf,
        leaf_fields,
        row_leaf,
        leaf_context,
    ):
        if type(max_depth) is not int or max_depth < 0:
            raise ValueError("nonnegative integer max_depth required")
        if max_leaves is None:
            max_leaves = len(data.data.values)
        if type(max_leaves) is not int or not 1 <= max_leaves <= 2**30:
            raise ValueError("positive bounded integer max_leaves required")
        if (row_leaf is None) != (leaf_context is None):
            raise ValueError("routed leaf solver and context must be supplied together")
        if row_leaf is not None and (
            leaf is not ops.newton_leaf
            or not callable(row_leaf)
            or not isinstance(leaf_context, ResidualContext)
            or leaf_context.problem.identity != fields.problem_identity
        ):
            raise ValueError(
                "routed leaf context must match problem; additive leaf must be default"
            )
        self.row_leaf, self.leaf_context = row_leaf, leaf_context
        self.data, self.fields = data, fields
        self.leaf_fields = fields if leaf_fields is None else leaf_fields
        if self.leaf_fields.problem_identity != fields.problem_identity:
            raise ValueError("split and leaf fields must belong to the same problem")
        ops.histogram(data, fields, [])  # Validate split identity even for a root-only tree.
        self.max_depth, self.max_leaves = max_depth, max_leaves
        self.scoring, self.legality, self.leaf = scoring, legality, leaf
        self.nodes, self.memberships, self.depths = [], [], []
        self.append(None, 0)

    def append(self, rows, depth):
        hist = ops.histogram(self.data, self.leaf_fields, rows)
        value = np.atleast_1d(
            np.asarray(
                self.leaf(hist.total, self.leaf_fields.names)
                if self.row_leaf is None
                else self.row_leaf(
                    self.leaf_context.view(hist.rows), hist.total, self.leaf_fields.names
                ),
                dtype=float,
            )
        )
        if value.ndim != 1 or not value.size or not np.isfinite(value).all():
            raise ValueError("nonfinite or invalid custom leaf")
        value = _owned(value, ndim=1)
        if self.nodes and value.shape != self.nodes[0][-1].shape:
            raise ValueError("inconsistent learner output width")
        self.nodes.append([-1, -1, False, -1, -1, value])
        self.memberships.append(hist.rows)
        self.depths.append(depth)
        return len(self.nodes) - 1

    def options(self, node):
        return ops.candidates(ops.histogram(self.data, self.fields, self.memberships[node]))

    def best(self, node):
        if self.depths[node] >= self.max_depth:
            return None
        gains = {}

        def cached(candidate):
            gain = float(self.scoring(candidate))
            gains[candidate.key] = gain
            return gain

        candidate = ops.choose(self.options(node), scoring=cached, legality=self.legality)
        return None if candidate is None else (candidate, gains[candidate.key])

    def split(self, node, candidate):
        left_rows, right_rows = ops.partition(self.data, self.memberships[node], candidate)
        if not len(left_rows) or not len(right_rows):
            raise ValueError("custom legality admitted an empty child")
        left = self.append(left_rows, self.depths[node] + 1)
        right = self.append(right_rows, self.depths[node] + 1)
        self.nodes[node][:5] = [
            candidate.feature,
            candidate.threshold,
            candidate.missing_left,
            left,
            right,
        ]
        return left, right

    def finish(self):
        return Tree(self.data.binning, *zip(*self.nodes, strict=True))


def depthwise(
    data,
    fields,
    *,
    max_depth=2,
    max_leaves=None,
    scoring=ops.score,
    legality=ops.feasible,
    leaf=ops.newton_leaf,
    leaf_fields=None,
    row_leaf=None,
    leaf_context=None,
):
    """Layer growth; highest-gain splits win a binding within-layer leaf budget."""
    work = _Growth(
        data,
        fields,
        max_depth,
        max_leaves,
        scoring,
        legality,
        leaf,
        leaf_fields,
        row_leaf,
        leaf_context,
    )
    frontier, leaves = [0], 1
    while frontier and leaves < work.max_leaves:
        choices = []
        for node in frontier:
            best = work.best(node)
            if best is not None:
                candidate, gain = best
                choices.append((node, candidate, gain))
        chosen = sorted(choices, key=lambda c: (-c[2], c[0], c[1].key))[: work.max_leaves - leaves]
        frontier = []
        for node, candidate, _gain in sorted(chosen, key=lambda c: c[0]):
            frontier.extend(work.split(node, candidate))
            leaves += 1
    return work.finish()


def best_first(
    data,
    fields,
    *,
    max_depth=2,
    max_leaves=None,
    scoring=ops.score,
    legality=ops.feasible,
    leaf=ops.newton_leaf,
    leaf_fields=None,
    row_leaf=None,
    leaf_context=None,
):
    """Heap of leaf gains; unchanged leaves retain their evaluated candidates.

    Ties use node ID then condition. Pure scoring/legality callbacks must depend
    on their candidate and immutable configuration, not an evolving call count.
    """
    work = _Growth(
        data,
        fields,
        max_depth,
        max_leaves,
        scoring,
        legality,
        leaf,
        leaf_fields,
        row_leaf,
        leaf_context,
    )
    pending = []

    def enqueue(node):
        best = work.best(node)
        if best is not None:
            candidate, gain = best
            heapq.heappush(pending, (-gain, node, candidate.key, candidate))

    leaves = 1
    if leaves < work.max_leaves:
        enqueue(0)
    while pending and leaves < work.max_leaves:
        _gain, node, _key, candidate = heapq.heappop(pending)
        children = work.split(node, candidate)
        leaves += 1
        if leaves < work.max_leaves:
            for child in children:
                enqueue(child)
    return work.finish()


def symmetric(
    data,
    fields,
    *,
    max_depth=2,
    max_leaves=None,
    scoring=ops.score,
    legality=ops.feasible,
    leaf=ops.newton_leaf,
    leaf_fields=None,
    row_leaf=None,
    leaf_context=None,
):
    """One common condition per complete layer, legal in every active leaf.

    Sum gains for each common condition, including negative per-node gains.
    Split only for a positive total and a budget permitting the entire layer.
    """
    work = _Growth(
        data,
        fields,
        max_depth,
        max_leaves,
        scoring,
        legality,
        leaf,
        leaf_fields,
        row_leaf,
        leaf_context,
    )
    frontier = [0]
    for _ in range(max_depth):
        if 2 * len(frontier) > work.max_leaves:
            break
        by_node = []
        for node in frontier:
            candidates = {}
            for candidate in work.options(node):
                if legality(candidate):
                    gain = float(scoring(candidate))
                    if not np.isfinite(gain):
                        raise ValueError("nonfinite custom split score")
                    candidates[candidate.key] = (candidate, gain)
            by_node.append(candidates)
        common = set.intersection(*(set(options) for options in by_node))
        if not common:
            break
        totals = {key: sum(options[key][1] for options in by_node) for key in common}
        if not all(np.isfinite(value) for value in totals.values()):
            raise ValueError("nonfinite symmetric layer score")
        key = min(common, key=lambda k: (-totals[k], k))
        if totals[key] <= 0:
            break
        next_frontier = []
        for node, options in zip(frontier, by_node, strict=True):
            next_frontier.extend(work.split(node, options[key][0]))
        frontier = next_frontier
    return work.finish()
