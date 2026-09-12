"""Exact CPU scalar Newton ordering over caller-prepared numeric/category bins.

Integer bin sums preserve every stored binary64 field bit. Rational gains and
feasibility comparisons have no tolerance band. This CPU operation is explicit;
existing tree/recipe defaults and arbitrary scoring callbacks are unchanged.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .binning import BinnedData
from .data import _identity, _owned
from .ops import Candidate, _nonnegative, _rows
from .stats import RowFields


@dataclass(frozen=True)
class NewtonCandidate:
    """Routing candidate, exact gain, and exact (left, right, parent) field sums.

    exact_sums follow candidate.names. Candidate arrays are correctly rounded
    finite binary64 sums for ordinary routing/consumer interoperability. Compare
    gain directly; casting it to float discards the ordering guarantee.
    """

    candidate: Candidate
    gain: Fraction
    exact_sums: tuple[tuple[Fraction, ...], tuple[Fraction, ...], tuple[Fraction, ...]]
    evaluation_identity: str
    input_identity: str
    configuration_identity: str


def _input_identity(data, fields, rows):
    return _identity(
        "exact-newton-input-v1",
        data.identity,
        fields.problem_identity,
        fields.names,
        fields.roles,
        fields.values,
        rows,
    )


def _parameter(value):
    try:
        return Fraction(_nonnegative(value))
    except (TypeError, OverflowError) as error:
        raise ValueError("finite nonnegative binary64 parameter required") from error


def _integers(values):
    """One exact dyadic scale per column; only integer additions follow."""
    ratios = [[float(v).as_integer_ratio() for v in row] for row in values]
    powers = [
        max((row[q][1].bit_length() - 1 for row in ratios), default=0)
        for q in range(values.shape[1])
    ]
    integers = [
        tuple(n << (powers[q] - (d.bit_length() - 1)) for q, (n, d) in enumerate(row))
        for row in ratios
    ]
    return integers, tuple(1 << p for p in powers)


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b, strict=True))


def _subtract(a, b):
    return tuple(x - y for x, y in zip(a, b, strict=True))


def _floating(sums):
    try:
        return _owned([float(v) for v in sums], ndim=1)
    except OverflowError as error:
        raise ValueError("exact aggregate is not representable in a routing candidate") from error


def _columns(fields):
    if not isinstance(fields, RowFields):
        raise ValueError("original row fields required")
    if "unweighted" in fields.roles:
        raise ValueError("apply objective training weights before Newton operations")
    if "gradient" not in fields.names or "curvature" not in fields.names:
        raise ValueError("named scalar gradient and curvature required")
    g, h = (fields.names.index(n) for n in ("gradient", "curvature"))
    if (
        fields.roles[g] != "training"
        or fields.roles[h] != "training"
        or np.any(fields.values[:, h] < 0)
    ):
        raise ValueError("once-weighted training G/H and nonnegative row curvature required")
    return g, h


def leaf(fields, rows=None, *, reg_lambda=1.0):
    """Round -sum(G)/(sum(H)+lambda) once after exact original-row reduction.

    Named scalar G/H must already include training weights. Intermediate exact
    sums may exceed binary64; only the returned solution must be finite. Invalid
    or zero denominators fail explicitly, without clipping or reweighting.
    """
    g, h = _columns(fields)
    regularization = _parameter(reg_lambda)
    selected = _rows(rows, len(fields.values))
    values, scales = _integers(fields.values[selected][:, [g, h]])
    total = (0, 0)
    for row in values:
        total = _add(total, row)
    numerator = Fraction(total[0], scales[0])
    denominator = Fraction(total[1], scales[1]) + regularization
    if denominator <= 0:
        raise ValueError("positive exact Newton denominator required")
    try:
        value = float(-numerator / denominator)
    except OverflowError as error:
        raise ValueError("nonfinite exact Newton leaf") from error
    return value


def rank(
    data,
    fields,
    rows=None,
    *,
    reg_lambda=1.0,
    split_penalty=0.0,
    min_child_h=0.0,
    min_information=None,
):
    """Return all exactly feasible scalar Newton candidates in descending order.

    Strict positive curvature, row counts and named independent minima define
    feasibility. Legal zero/negative gains remain available for layer sums.
    Both missing routes and categorical equality conditions follow BinnedData;
    missing and unknown categories retain the prepared encoding's semantics.

    G/H must be once-weighted training fields with nonnegative row curvature.
    Independent fields are summed without objective weights. No data fitting,
    device execution, approximate score filtering or CPU/GPU fallback occurs.
    Cost of this exact CPU baseline is not a performance guarantee.
    """
    if (
        not isinstance(data, BinnedData)
        or not isinstance(fields, RowFields)
        or fields.data_identity != data.data.identity
        or len(fields.values) != len(data.data.values)
    ):
        raise ValueError("prepared data and fields must share original row identity")
    g, h = _columns(fields)
    regularization, penalty, minimum = (
        _parameter(v) for v in (reg_lambda, split_penalty, min_child_h)
    )
    if min_information is not None and not isinstance(min_information, Mapping):
        raise ValueError("named independent minima mapping required")
    information = []
    for name, value in (min_information or {}).items():
        if name not in fields.names or fields.roles[fields.names.index(name)] != "independent":
            raise ValueError("information minima require named independent fields")
        information.append((fields.names.index(name), _parameter(value)))
    selected = _rows(rows, len(data.data.values))
    values, scales = _integers(fields.values[selected])
    width = len(fields.names)
    zero = (0,) * width
    total = zero
    for row in values:
        total = _add(total, row)

    def rational(sums):
        return tuple(Fraction(n, d) for n, d in zip(sums, scales, strict=True))

    parent = rational(total)
    parent_float = _floating(parent)
    row_identity = _identity(selected)
    input_identity = _input_identity(data, fields, selected)
    configuration_identity = _identity(
        "exact-newton-configuration-v1",
        str(regularization),
        str(penalty),
        str(minimum),
        tuple((q, str(v)) for q, v in sorted(information)),
    )
    evaluation_identity = _identity(input_identity, configuration_identity)

    def node(sums):
        return sums[g] ** 2 / (2 * (sums[h] + regularization))

    result = []
    for feature, bins in enumerate(data.binning.bin_counts):
        sums, counts = [zero] * (bins + 1), [0] * (bins + 1)
        for i, row in enumerate(selected):
            code = bins if data.missing[feature, row] else int(data.codes[feature, row])
            sums[code] = _add(sums[code], values[i])
            counts[code] += 1
        regular = _subtract(total, sums[-1])
        regular_count = len(selected) - counts[-1]
        prefixes, prefix_counts = [], []
        running, count = zero, 0
        for cell, n in zip(sums[:-1], counts[:-1], strict=True):
            running, count = _add(running, cell), count + n
            prefixes.append(running)
            prefix_counts.append(count)
        active = np.unique(data.codes[feature, ~data.missing[feature]])
        categorical = data.binning.categories[feature] is not None
        for threshold in active:
            left_regular = sums[threshold] if categorical else prefixes[threshold]
            left_count = counts[threshold] if categorical else prefix_counts[threshold]
            right_regular = _subtract(regular, left_regular)
            for missing_left in (False, True):
                left = rational(_add(left_regular, sums[-1]) if missing_left else left_regular)
                right = rational(right_regular if missing_left else _add(right_regular, sums[-1]))
                nl = left_count + (counts[-1] if missing_left else 0)
                nr = regular_count - left_count + (0 if missing_left else counts[-1])
                if (
                    min(nl, nr) <= 0
                    or min(left[h], right[h]) <= 0
                    or min(left[h], right[h]) < minimum
                    or any(min(left[q], right[q]) < value for q, value in information)
                ):
                    continue
                gain = node(left) + node(right) - node(parent) - penalty
                candidate = Candidate(
                    feature,
                    int(threshold),
                    missing_left,
                    fields.names,
                    fields.roles,
                    _floating(left),
                    _floating(right),
                    parent_float,
                    nl,
                    nr,
                    data.identity,
                    row_identity,
                    "categorical" if categorical else "numeric",
                )
                result.append(
                    NewtonCandidate(
                        candidate,
                        gain,
                        (left, right, parent),
                        evaluation_identity,
                        input_identity,
                        configuration_identity,
                    )
                )
    return tuple(sorted(result, key=lambda r: (-r.gain, r.candidate.key)))


def choose(ranked):
    """Select a strictly positive exact gain with lexicographic condition ties.

    A caller may filter rank's records using its own additional constraints.
    Records from different nodes or duplicate conditions cannot be mixed.
    """
    ranked = tuple(ranked)
    if any(
        not isinstance(r, NewtonCandidate)
        or not isinstance(r.candidate, Candidate)
        or not isinstance(r.gain, Fraction)
        or not isinstance(r.evaluation_identity, str)
        or not r.evaluation_identity
        or not isinstance(r.input_identity, str)
        or not r.input_identity
        or not isinstance(r.configuration_identity, str)
        or not r.configuration_identity
        or r.evaluation_identity != _identity(r.input_identity, r.configuration_identity)
        for r in ranked
    ):
        raise ValueError("exact Newton candidate records required")
    identities = {
        (
            r.evaluation_identity,
            r.candidate.data_identity,
            r.candidate.rows_identity,
            r.candidate.names,
            r.candidate.roles,
        )
        for r in ranked
    }
    if len(identities) > 1 or len({r.candidate.key for r in ranked}) != len(ranked):
        raise ValueError(
            "one objective/configuration binding, original-row node and unique conditions required"
        )
    positive = [r for r in ranked if r.gain > 0]
    return min(positive, key=lambda r: (-r.gain, r.candidate.key)) if positive else None
