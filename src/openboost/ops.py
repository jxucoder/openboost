"""Composable scalar CPU histogram, candidate, routing and Newton operations."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .binning import BinnedData, _array
from .data import _identity, _owned
from .stats import RowFields


def _rows(rows, size):
    a = np.arange(size) if rows is None else np.asarray(rows)
    if (
        a.ndim != 1
        or (a.size and a.dtype.kind not in "iu")
        or np.any(a < 0)
        or np.any(a >= size)
        or len(np.unique(a)) != len(a)
    ):
        raise ValueError("unique in-range row positions required")
    return _array(a, "<i8")


@dataclass(frozen=True, eq=False)
class Histogram:
    data: BinnedData
    rows: np.ndarray
    fields: RowFields
    sums: tuple[np.ndarray, ...]
    counts: tuple[np.ndarray, ...]
    total: np.ndarray


def histogram(data, fields, rows=None):
    """Aggregate exactly the selected original rows, with one separate missing bin.

    Fields are additive. Unweighted objective fields must be adapted before this
    call; independent information is never multiplied by training weights here.
    """
    if (
        not isinstance(data, BinnedData)
        or not isinstance(fields, RowFields)
        or fields.data_identity != data.data.identity
        or len(fields.values) != len(data.data.values)
    ):
        raise ValueError("data and fields have different row identity")
    if "unweighted" in fields.roles:
        raise ValueError("apply objective training weights before aggregation")
    selected = _rows(rows, len(data.data.values))
    selected_values = fields.values[selected]
    # Preserve the original C-order reduction before laying out contiguous columns.
    total = _owned(selected_values.sum(axis=0), ndim=1)
    columns = np.ascontiguousarray(selected_values.T)
    del selected_values
    sums, counts = [], []
    for feature, bins in enumerate(data.binning.bin_counts):
        codes = np.where(data.missing[feature, selected], bins, data.codes[feature, selected])
        sums.append(
            _owned(
                np.column_stack(
                    [np.bincount(codes, weights=column, minlength=bins + 1) for column in columns]
                ),
                ndim=2,
            )
        )
        counts.append(_array(np.bincount(codes, minlength=bins + 1), "<i8"))
    return Histogram(
        data,
        selected,
        fields,
        tuple(sums),
        tuple(counts),
        total,
    )


@dataclass(frozen=True, eq=False)
class Candidate:
    feature: int
    threshold: int
    missing_left: bool
    names: tuple[str, ...]
    roles: tuple[str, ...]
    left: np.ndarray
    right: np.ndarray
    parent: np.ndarray
    left_count: int
    right_count: int
    data_identity: str
    rows_identity: str
    kind: str = "numeric"

    @property
    def key(self):
        return self.feature, self.threshold, self.missing_left


def candidates(hist):
    """Prefix/suffix sums; enumerate both missing routes, including missing-only splits."""
    result = []
    rows_identity = _identity(hist.rows)
    for feature, (sums, counts) in enumerate(zip(hist.sums, hist.counts, strict=True)):
        prefix = np.cumsum(sums[:-1], axis=0)
        suffix = np.cumsum(sums[:-1][::-1], axis=0)[::-1]
        prefix_count = np.cumsum(counts[:-1])
        active = np.unique(hist.data.codes[feature, ~hist.data.missing[feature]])
        categorical = hist.data.binning.categories[feature] is not None
        for threshold in active:
            for missing_left in (False, True):
                regular_left = sums[threshold] if categorical else prefix[threshold]
                regular_right = (
                    sums[:threshold].sum(axis=0) + sums[threshold + 1 : -1].sum(axis=0)
                    if categorical
                    else suffix[threshold + 1]
                    if threshold + 1 < len(suffix)
                    else np.zeros_like(regular_left)
                )
                left = regular_left + (sums[-1] if missing_left else 0)
                right = regular_right + (0 if missing_left else sums[-1])
                count = counts[threshold] if categorical else prefix_count[threshold]
                nleft = int(count + (counts[-1] if missing_left else 0))
                result.append(
                    Candidate(
                        feature,
                        int(threshold),
                        missing_left,
                        hist.fields.names,
                        hist.fields.roles,
                        _owned(left, ndim=1),
                        _owned(right, ndim=1),
                        hist.total,
                        nleft,
                        len(hist.rows) - nleft,
                        hist.data.identity,
                        rows_identity,
                        "categorical" if categorical else "numeric",
                    )
                )
    return tuple(result)


def _nonnegative(value):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError("finite nonnegative parameter required")
    return float(value)


def newton_leaf(total, names, *, reg_lambda=1.0):
    """Scalar Newton leaf from already-weighted sums, with half-square scoring."""
    g, h = total[names.index("gradient")], total[names.index("curvature")]
    denominator = _nonnegative(h) + _nonnegative(reg_lambda)
    return _newton_value(g, denominator)


def _newton_value(g, denominator):
    if denominator <= 0 or not np.isfinite(denominator) or not np.isfinite(g):
        raise ValueError("positive finite Newton denominator and finite gradient required")
    value = -float(g) / denominator
    if not np.isfinite(value):
        raise ValueError("nonfinite Newton leaf")
    return value


def feasible(candidate, *, min_child_h=0.0, min_information=None):
    """Default Newton legality plus optional named independent information minima."""
    minimum = _nonnegative(min_child_h)
    h = candidate.names.index("curvature")
    if (
        min(candidate.left_count, candidate.right_count) <= 0
        or min(candidate.left[h], candidate.right[h]) <= 0
        or min(candidate.left[h], candidate.right[h]) < minimum
    ):
        return False
    for name, value in (min_information or {}).items():
        i = candidate.names.index(name)
        if candidate.roles[i] != "independent":
            raise ValueError("information minima require independent fields")
        if min(candidate.left[i], candidate.right[i]) < _nonnegative(value):
            return False
    return True


def score(candidate, *, reg_lambda=1.0, split_penalty=0.0):
    def node(total):
        return (
            -0.5
            * total[candidate.names.index("gradient")]
            * newton_leaf(total, candidate.names, reg_lambda=reg_lambda)
        )

    gain = (
        node(candidate.left)
        + node(candidate.right)
        - node(candidate.parent)
        - _nonnegative(split_penalty)
    )
    if not np.isfinite(gain):
        raise ValueError("nonfinite split gain")
    return float(gain)


def choose(options, *, scoring=score, legality=feasible):
    """Public callbacks; highest strictly positive gain, lexicographic condition ties."""
    best, best_gain = None, 0.0
    for candidate in options:
        if not legality(candidate):
            continue
        gain = float(scoring(candidate))
        if not np.isfinite(gain):
            raise ValueError("nonfinite custom split score")
        if gain > best_gain or (
            gain == best_gain and best is not None and candidate.key < best.key
        ):
            best, best_gain = candidate, gain
    return best


def newton_choice(
    hist, *, reg_lambda=1.0, min_child_h=0.0, split_penalty=0.0, min_information=None
):
    """Return ``(Candidate, gain)`` or None using ordinary rounded Newton rules.

    Numeric histograms use array operations with the same prefix/suffix and
    half-square evaluation order as candidates/feasible/score/choose. Exact
    floating ties use the lexicographic condition key; no tolerance is applied.
    Categorical conditions, information constraints and exceptional arithmetic
    use those public operations directly. Returned fields own immutable bytes.
    This operation does not implement the separate exact-rational ordering policy.
    """

    def exhaustive():
        best = choose(
            candidates(hist),
            scoring=lambda c: score(c, reg_lambda=reg_lambda, split_penalty=split_penalty),
            legality=lambda c: feasible(
                c, min_child_h=min_child_h, min_information=min_information
            ),
        )
        return (
            None
            if best is None
            else (best, score(best, reg_lambda=reg_lambda, split_penalty=split_penalty))
        )

    if min_information or any(c is not None for c in hist.data.binning.categories):
        return exhaustive()
    try:
        regularizer = _nonnegative(reg_lambda)
        minimum = _nonnegative(min_child_h)
        penalty = _nonnegative(split_penalty)
        g, h = hist.fields.names.index("gradient"), hist.fields.names.index("curvature")
    except (ValueError, TypeError):
        return exhaustive()

    # Finish all candidate-field construction before scoring, preserving the
    # exhaustive path's rejection of invalid statistics even for illegal splits.
    blocks = []
    for feature, (sums, counts) in enumerate(zip(hist.sums, hist.counts, strict=True)):
        active = np.unique(hist.data.codes[feature, ~hist.data.missing[feature]])
        if not len(active):
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            prefix = np.cumsum(sums[:-1], axis=0)[active]
            suffix = np.concatenate(
                (np.cumsum(sums[:-1][::-1], axis=0)[::-1], np.zeros_like(sums[:1])), axis=0
            )[active + 1]
            left = np.empty((2 * len(active), sums.shape[1]), dtype=np.float64)
            right = np.empty_like(left)
            left[0::2], left[1::2] = prefix + 0, prefix + sums[-1]
            right[0::2], right[1::2] = suffix + sums[-1], suffix + 0
        if not np.isfinite(left).all() or not np.isfinite(right).all():
            return exhaustive()
        count = np.cumsum(counts[:-1])[active]
        nleft = np.empty(2 * len(active), dtype=np.int64)
        nleft[0::2], nleft[1::2] = count, count + counts[-1]
        legal = (
            (nleft > 0)
            & (nleft < len(hist.rows))
            & (left[:, h] > 0)
            & (right[:, h] > 0)
            & (left[:, h] >= minimum)
            & (right[:, h] >= minimum)
        )
        blocks.append((feature, active, left, right, nleft, legal))

    best, best_gain = None, 0.0
    parent = None
    for feature, active, left, right, nleft, legal in blocks:
        positions = np.flatnonzero(legal)
        if not len(positions):
            continue
        if parent is None:
            # Parent validation and arithmetic occur only when a legal split exists.
            try:
                parent = (
                    -0.5
                    * hist.total[g]
                    * newton_leaf(hist.total, hist.fields.names, reg_lambda=regularizer)
                )
            except ValueError:
                return exhaustive()
        a, b = left[positions], right[positions]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ad, bd = a[:, h] + regularizer, b[:, h] + regularizer
            av, bv = -a[:, g] / ad, -b[:, g] / bd
            gain = ((-0.5 * a[:, g]) * av + (-0.5 * b[:, g]) * bv) - parent - penalty
        if not all(np.isfinite(v).all() for v in (ad, bd, av, bv, gain)):
            return exhaustive()
        slot = int(np.argmax(gain))
        value = float(gain[slot])
        if value > best_gain:
            position = int(positions[slot])
            best_gain = value
            best = (
                feature,
                int(active[position // 2]),
                bool(position % 2),
                left[position],
                right[position],
                int(nleft[position]),
            )
    if best is None:
        return None
    feature, threshold, missing_left, left, right, count = best
    return Candidate(
        feature,
        threshold,
        missing_left,
        hist.fields.names,
        hist.fields.roles,
        _owned(left, ndim=1),
        _owned(right, ndim=1),
        hist.total,
        count,
        len(hist.rows) - count,
        hist.data.identity,
        _identity(hist.rows),
    ), best_gain


def partition(data, rows, candidate):
    """Return original positional indices; row IDs remain available on data.data."""
    selected = _rows(rows, len(data.data.values))
    if candidate.data_identity != data.identity or candidate.rows_identity != _identity(selected):
        raise ValueError("candidate belongs to different data or routed rows")
    if candidate.kind != data.binning.feature_kinds[candidate.feature]:
        raise ValueError("candidate condition kind differs from transformer")
    mask = np.where(
        data.missing[candidate.feature, selected],
        candidate.missing_left,
        data.codes[candidate.feature, selected] == candidate.threshold
        if candidate.kind == "categorical"
        else data.codes[candidate.feature, selected] <= candidate.threshold,
    )
    return _array(selected[mask], "<i8"), _array(selected[~mask], "<i8")


def _bound_choice(hist, result):
    """Bind a cooperative selector's single result to actual histogram fields."""
    if result is None:
        return None
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("selection must return a Candidate/gain pair or None")
    candidate, gain = result
    if not isinstance(candidate, Candidate):
        raise ValueError("selection must return a Candidate")
    f, t, missing = candidate.feature, candidate.threshold, candidate.missing_left
    if (
        type(f) is not int
        or not 0 <= f < len(hist.sums)
        or type(t) is not int
        or type(missing) is not bool
        or candidate.data_identity != hist.data.identity
        or candidate.rows_identity != _identity(hist.rows)
        or candidate.names != hist.fields.names
        or candidate.roles != hist.fields.roles
    ):
        raise ValueError("selection candidate must bind the actual node and fields")
    active = np.unique(hist.data.codes[f, ~hist.data.missing[f]])
    if t not in active or candidate.kind != hist.data.binning.feature_kinds[f]:
        raise ValueError("selection condition must belong to the actual histogram")
    sums, counts = hist.sums[f], hist.counts[f]
    if candidate.kind == "categorical":
        left = sums[t]
        right = sums[:t].sum(axis=0) + sums[t + 1 : -1].sum(axis=0)
        count = counts[t]
    else:
        left = np.cumsum(sums[:-1], axis=0)[t]
        suffix = np.cumsum(sums[:-1][::-1], axis=0)[::-1]
        right = suffix[t + 1] if t + 1 < len(suffix) else np.zeros_like(left)
        count = np.cumsum(counts[:-1])[t]
    left = _owned(left + (sums[-1] if missing else 0), ndim=1)
    right = _owned(right + (0 if missing else sums[-1]), ndim=1)
    count = int(count + (counts[-1] if missing else 0))
    for value, expected in (
        (candidate.left, left),
        (candidate.right, right),
        (candidate.parent, hist.total),
    ):
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != expected.dtype
            or value.shape != expected.shape
            or value.tobytes() != expected.tobytes()
        ):
            raise ValueError("selection fields differ from the actual histogram")
    if (
        type(candidate.left_count) is not int
        or type(candidate.right_count) is not int
        or candidate.left_count != count
        or candidate.right_count != len(hist.rows) - count
        or min(count, len(hist.rows) - count) <= 0
    ):
        raise ValueError("selection counts must match nonempty actual children")
    if not np.isscalar(gain) or not np.isfinite(gain) or gain <= 0:
        raise ValueError("selection requires a finite strictly positive gain")
    # The callback's buffers remain borrowed; growth uses canonical owned fields.
    return Candidate(
        f,
        t,
        missing,
        hist.fields.names,
        hist.fields.roles,
        left,
        right,
        hist.total,
        count,
        len(hist.rows) - count,
        hist.data.identity,
        candidate.rows_identity,
        candidate.kind,
    ), float(gain)


def _vector_indices(names):
    # Return caller-owned lists: cached mutable indices could poison later trees.
    key = tuple(names)
    if not all(isinstance(name, str) for name in key):
        layout = _vector_layout.__wrapped__(key)
    else:
        layout = _vector_layout(key)
    return list(layout[0]), list(layout[1])


@lru_cache(maxsize=128)
def _vector_layout(names):
    """Bounded immutable schema metadata; no data, gradients, or run state."""
    width = sum(n.startswith("gradient:") for n in names)
    if width == 0:
        raise ValueError("vector Newton fields required")
    return (
        tuple(names.index(f"gradient:{k}") for k in range(width)),
        tuple(names.index(f"curvature:{k}") for k in range(width)),
    )


def vector_leaf(total, names, *, reg_lambda=1.0):
    g, h = _vector_indices(names)
    return _owned(
        [
            newton_leaf([total[i], total[j]], ("gradient", "curvature"), reg_lambda=reg_lambda)
            for i, j in zip(g, h, strict=True)
        ],
        ndim=1,
    )


def vector_score(candidate, *, reg_lambda=1.0, split_penalty=0.0):
    g, h = _vector_indices(candidate.names)
    regularizer = _nonnegative(reg_lambda)

    def node(total):
        with np.errstate(over="raise", invalid="raise"):
            # Scratch values never escape scoring; persisted leaves remain owned.
            values = np.fromiter(
                (
                    _newton_value(total[i], _nonnegative(total[j]) + regularizer)
                    for i, j in zip(g, h, strict=True)
                ),
                dtype=np.float64,
                count=len(g),
            )
            return -0.5 * np.dot(total[g], values)

    gain = (
        node(candidate.left)
        + node(candidate.right)
        - node(candidate.parent)
        - _nonnegative(split_penalty)
    )
    if not np.isfinite(gain):
        raise ValueError("nonfinite vector split score")
    return float(gain)


def vector_feasible(candidate, *, min_child_h=0.0):
    _g, h = _vector_indices(candidate.names)
    minimum = _nonnegative(min_child_h)
    return (
        min(candidate.left_count, candidate.right_count) > 0
        and np.all(candidate.left[h] > 0)
        and np.all(candidate.right[h] > 0)
        and np.all(candidate.left[h] >= minimum)
        and np.all(candidate.right[h] >= minimum)
    )
