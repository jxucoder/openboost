"""Compare public histogram operations with independent row-enumeration oracles."""

from functools import partial

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost.binning import Binning
from openboost.ops import candidates, choose, feasible, histogram, newton_leaf, partition, score
from openboost.stats import apply_weight, newton
from tests.v1.reference.data import NumericBinning as ReferenceBinning
from tests.v1.reference.scalar import newton_leaf as reference_leaf
from tests.v1.reference.tree import best_split, enumerate_splits


def setup(values, weight=None):
    x = np.asarray(values, dtype=float)
    data = NumericData(x, np.arange(len(x)) + 100, tuple(f"f{i}" for i in range(x.shape[1])))
    problem = Problem(data, np.zeros((len(x), 1)), data.row_ids, weight=weight)
    binned = Binning.fit(data, bins=4).transform(data)
    return problem, binned


@pytest.mark.parametrize(
    "column,bins",
    [([0, 0, 0, 1], 2), ([np.nan, np.nan], 4), ([3, 3, 3], 5), ([-4, -1, 2, 9], 4), ([1, 3], 1)],
)
def test_binning_matches_order_statistic_reference(column, bins):
    data = NumericData(np.asarray(column)[:, None], np.arange(len(column)), ("x",))
    fitted = Binning.fit(data, bins=bins)
    ref = ReferenceBinning.fit(column, bins=bins)
    np.testing.assert_allclose(fitted.cuts[0], ref.cuts)
    target = NumericData([[-100], [0], [100], [np.nan]], [1, 2, 3, 4], ("x",))
    got = fitted.transform(target)
    codes, missing = ref.transform(target.values[:, 0])
    np.testing.assert_array_equal(got.codes[0], codes)
    np.testing.assert_array_equal(got.missing[0], missing)
    assert got.codes.dtype == np.int32 and got.codes.shape == (1, 4)
    with pytest.raises(ValueError):
        got.codes.flags.writeable = True


@pytest.mark.parametrize("rows", [None, [0, 2, 4, 5], []])
def test_histogram_candidates_and_routing_match_exhaustive_oracle(rows):
    p, b = setup(
        [[0, 1], [0, np.nan], [1, 2], [2, 0], [np.nan, 1], [3, 3]], weight=[1, 0, 2, 1, 3, 1]
    )
    g, h = np.array([-6, 1, 1, 1, 1, 2.0]), np.ones(6)
    info = np.eye(2)[np.arange(6) % 2]
    fields = newton(p, g, h).add_independent("a", info[:, 0]).add_independent("b", info[:, 1])
    hist = histogram(b, fields, rows)
    selected = np.arange(6) if rows is None else np.asarray(rows, dtype=int)
    np.testing.assert_allclose(
        hist.total[:2], [np.dot(p.weight[selected], g[selected]), p.weight[selected].sum()]
    )
    for counts, sums in zip(hist.counts, hist.sums, strict=True):
        assert counts.sum() == len(selected)
        np.testing.assert_allclose(sums.sum(axis=0), hist.total)
    bins = np.where(b.missing.T, np.nan, b.codes.T)
    ref = enumerate_splits(
        bins, g, h, weight=p.weight, rows=selected, information=info, min_information=1
    )
    options = candidates(hist)
    legal = partial(feasible, min_information={"a": 1, "b": 1})
    observed = {c.key: c for c in options if legal(c)}
    assert set(observed) == {
        (c.condition.feature, c.condition.threshold, c.condition.missing_left) for c in ref
    }
    for c in ref:
        got = observed[(c.condition.feature, c.condition.threshold, c.condition.missing_left)]
        assert score(got) == pytest.approx(c.gain, abs=1e-12)
        left, right = partition(b, selected, got)
        np.testing.assert_array_equal(left, c.left)
        np.testing.assert_array_equal(right, c.right)
        assert newton_leaf(got.left, got.names) == pytest.approx(
            reference_leaf(got.left[0], got.left[1])
        )
    best, expected = choose(options, legality=legal), best_split(ref)
    assert (best is None) == (expected is None)
    if best:
        assert best.key == (
            expected.condition.feature,
            expected.condition.threshold,
            expected.condition.missing_left,
        )


def test_weight_once_independent_mass_and_foreign_identity():
    p, b = setup([[0], [1], [2]], [0, 2, 3])
    fields = newton(p, [-1, 2, 3], [1, 1, 1]).add_independent("cohort", [1, 1, 1])
    np.testing.assert_array_equal(histogram(b, fields).total, [13, 5, 3])
    with pytest.raises(ValueError, match="already applied"):
        apply_weight(fields, p)
    other = NumericData(p.data.values, p.row_ids[::-1], p.data.feature_names)
    with pytest.raises(ValueError, match="identity"):
        histogram(b.binning.transform(other), fields)


def test_constraint_changes_winner_through_public_callback():
    p, b = setup(np.arange(6)[:, None])
    b = Binning.fit(p.data, bins=6).transform(p.data)
    fields = newton(p, [-6, 1, 1, 1, 1, 2], np.ones(6))
    fields = fields.add_independent("a", [1, 0, 1, 0, 1, 0]).add_independent(
        "b", [0, 1, 0, 1, 0, 1]
    )
    options = candidates(histogram(b, fields))
    unconstrained = choose(options)
    constrained = choose(options, legality=partial(feasible, min_information={"a": 1, "b": 1}))
    assert unconstrained is not None and constrained is not None
    assert unconstrained.key != constrained.key
    left, right = partition(b, None, constrained)
    for rows in (left, right):
        assert fields.values[rows, 2:].sum(axis=0).min() >= 1
    with pytest.raises(ValueError, match="routed rows"):
        partition(b, [0, 1], constrained)
    with pytest.raises(ValueError, match="nonfinite"):
        choose(options, scoring=lambda _: np.nan)


def test_two_level_operations_keep_original_row_positions():
    p, b = setup(np.arange(8)[:, None])
    fields = newton(p, [-4, -4, -1, -1, 1, 1, 4, 4], np.ones(8))
    root = choose(candidates(histogram(b, fields)))
    children = partition(b, None, root)
    leaves = []
    for rows in children:
        child = choose(candidates(histogram(b, fields, rows)))
        leaves.extend(partition(b, rows, child) if child else [rows])
    np.testing.assert_array_equal(np.sort(np.concatenate(leaves)), np.arange(8))
    assert len(leaves) == 4
    for rows in leaves:
        total = histogram(b, fields, rows).total
        expected = reference_leaf(fields.values[rows, 0].sum(), fields.values[rows, 1].sum())
        assert newton_leaf(total, fields.names) == expected


def test_missing_only_split_and_all_missing_feature():
    p, b = setup([[1, np.nan], [1, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    fields = newton(p, [-2, -2, 2, 2], np.ones(4))
    options = candidates(histogram(b, fields))
    assert all(c.feature == 0 for c in options)
    chosen = choose(options)
    left, right = partition(b, None, chosen)
    np.testing.assert_array_equal(left, [0, 1])
    np.testing.assert_array_equal(right, [2, 3])
    assert chosen.key == (0, 0, False)


@pytest.mark.parametrize("rows", [[0, 0], [-1], [4], [1.5]])
def test_invalid_routes_are_not_coerced(rows):
    p, b = setup([[0], [1], [2]])
    with pytest.raises(ValueError, match="row positions"):
        histogram(b, newton(p, [-1, 0, 1], np.ones(3)), rows)


def test_binning_identity_and_weight_role_cannot_be_reused_silently():
    p, b = setup([[0], [1], [2], [3]])
    fields = newton(p, [-2, -1, 1, 2], np.ones(4))
    options = candidates(histogram(b, fields))
    chosen = choose(options)
    other = Binning.fit(p.data, bins=2).transform(p.data)
    with pytest.raises(ValueError, match="different data"):
        partition(other, None, chosen)
    with pytest.raises(ValueError, match="independent"):
        feasible(chosen, min_information={"curvature": 1})


def test_overflowing_quantiles_cannot_silently_remove_cuts():
    data = NumericData([[-1e308], [1e308]], [1, 2], ("x",))
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(ValueError, match="interpolated cuts"),
    ):
        Binning.fit(data, bins=2)
