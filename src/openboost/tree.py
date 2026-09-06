"""Composable CPU depthwise scalar trees with validated numeric inference state."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import ops
from .binning import NumericBinning, _array
from .data import _identity, _owned


def _indices(value):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iu" or np.any(a < -1) or np.any(a > 2**31 - 1):
        raise ValueError("int32 topology vector required")
    return _array(a, "<i4")


@dataclass(frozen=True, eq=False)
class NumericTree:
    """Owned explicit topology; leaf feature/threshold/children use -1 sentinels.

    Predictions are scalar learner outputs [N, 1], without a base or offset.
    Training row membership is deliberately absent from inference artifacts.
    """

    binning: NumericBinning
    feature: np.ndarray
    threshold: np.ndarray
    missing_left: np.ndarray
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray

    def __post_init__(self):
        if not isinstance(self.binning, NumericBinning):
            raise ValueError("numeric transformer required")
        for name in ("feature", "threshold", "left", "right"):
            object.__setattr__(self, name, _indices(getattr(self, name)))
        values = _owned(self.value, ndim=1)
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
                if f >= len(self.binning.cuts) or t < 0 or t > len(self.binning.cuts[f]):
                    raise ValueError("split exceeds transformer schema")
                pending.extend((right, left))
        if len(seen) != n:
            raise ValueError("unreachable tree nodes")

    @property
    def identity(self):
        return _identity(self.record())

    def predict(self, data):
        binned = self.binning.transform(data)
        result = np.empty((len(data.values), 1))
        pending = [(0, np.arange(len(data.values)))]
        while pending:
            i, rows = pending.pop()
            f = self.feature[i]
            if f == -1:
                result[rows, 0] = self.value[i]
            else:
                mask = np.where(
                    binned.missing[f, rows],
                    self.missing_left[i],
                    binned.codes[f, rows] <= self.threshold[i],
                )
                pending.extend(((self.left[i], rows[mask]), (self.right[i], rows[~mask])))
        return result

    def record(self):
        return dict(
            format="openboost-numeric-tree-v1",
            feature_names=list(self.binning.feature_names),
            cuts=[c.tolist() for c in self.binning.cuts],
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
        vectors = ("feature", "threshold", "missing_left", "left", "right", "value")
        if (
            not isinstance(record, dict)
            or set(record) != {"format", "feature_names", "cuts", *vectors}
            or record["format"] != "openboost-numeric-tree-v1"
            or any(
                not isinstance(record[name], list) for name in ("feature_names", "cuts", *vectors)
            )
            or any(not isinstance(c, list) for c in record["cuts"])
        ):
            raise ValueError("unsupported or corrupt tree artifact schema")
        return cls(
            NumericBinning(record["feature_names"], record["cuts"]),
            **{name: record[name] for name in vectors},
        )


def depthwise(
    data,
    fields,
    *,
    max_depth=2,
    max_leaves=None,
    scoring=ops.score,
    legality=ops.feasible,
    leaf=ops.newton_leaf,
):
    """Grow layers; under a leaf cap select the highest-gain splits in each layer.

    Callbacks receive weighted additive statistics. Scoring is evaluated once per
    legal candidate; the cached gain also ranks competing nodes within a layer.
    Leaf receives (total, field_names). No callback may silently create empty leaves.
    """
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("nonnegative integer max_depth required")
    if max_leaves is None:
        max_leaves = len(data.data.values)
    if type(max_leaves) is not int or not 1 <= max_leaves <= 2**30:
        raise ValueError("positive bounded integer max_leaves required")
    nodes, memberships = [], []

    def append(rows):
        hist = ops.histogram(data, fields, rows)
        value = float(leaf(hist.total, fields.names))
        if not np.isfinite(value):
            raise ValueError("nonfinite custom leaf")
        nodes.append([-1, -1, False, -1, -1, value])
        memberships.append(hist.rows)
        return len(nodes) - 1

    append(None)
    frontier, leaves = [0], 1
    for _ in range(max_depth):
        if leaves >= max_leaves:
            break
        choices = []
        for i in frontier:
            gains = {}

            def cached(candidate, gains=gains):
                gain = float(scoring(candidate))
                gains[candidate.key] = gain
                return gain

            best = ops.choose(
                ops.candidates(ops.histogram(data, fields, memberships[i])),
                scoring=cached,
                legality=legality,
            )
            if best is not None:
                choices.append((i, best, gains[best.key]))
        chosen = sorted(choices, key=lambda c: (-c[2], c[0], c[1].key))[: max_leaves - leaves]
        if not chosen:
            break
        frontier = []
        for i, candidate, _gain in sorted(chosen, key=lambda c: c[0]):
            left_rows, right_rows = ops.partition(data, memberships[i], candidate)
            if not len(left_rows) or not len(right_rows):
                raise ValueError("custom legality admitted an empty child")
            left, right = append(left_rows), append(right_rows)
            nodes[i][:5] = [
                candidate.feature,
                candidate.threshold,
                candidate.missing_left,
                left,
                right,
            ]
            frontier.extend((left, right))
            leaves += 1
    columns = list(zip(*nodes, strict=True))
    return NumericTree(data.binning, *columns)
