"""Exact arithmetic controls for the retained Normal structural counterexample."""

from fractions import Fraction

import pytest

from .reference.normal_split_order import candidates, captured, exact_tree, predict, winner


def test_captured_root_scores_are_exact_ties_in_both_storage_formats():
    for stored in (False, True):
        x, fields = captured(stored=stored)
        options = {r["key"]: r for r in candidates(x, fields, range(8))}
        a, b = (options[k] for k in ((0, 0, True), (0, 3, False)))
        assert a["gain"] == b["gain"] > 0
        assert a["sums"] == b["sums"][::-1]
        assert a["rows"] == ((0, 1, 6), (2, 3, 4, 5, 7))
        assert b["rows"] == ((0, 2, 3, 4, 5, 7), (1, 6))
        assert winner(options.values())["key"] == (0, 0, True)


@pytest.mark.parametrize("stored", [False, True])
def test_exact_tree_keeps_all_rows_and_lexicographic_choices(stored):
    x, fields = captured(stored=stored)
    nodes = exact_tree(x, fields, depth=2)
    assert [n["key"] for n in nodes] == [(0, 0, True), (1, 1, False), (0, 1, False), None, None, None, None]
    values = predict(nodes, x)
    assert values[0] == values[1] and values[0] != values[2]
    assert values[2] == values[3] and values[4] == values[5] == values[7]
    assert sorted(r for n in nodes if n["key"] is None for r in n["rows"]) == list(range(8))
    assert all(isinstance(v, Fraction) for v in values)


@pytest.mark.parametrize("power", [24, 54, 100, 149])
@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_tiny_positive_mass_is_a_genuine_non_tie_not_an_epsilon_tie(power, direction):
    x, fields = captured(stored=True)
    mass = Fraction(1, 2**power)
    fields[0] = (direction*mass, mass)
    options = {r["key"]: r for r in candidates(x, fields, range(8))}
    a, b = (options[k] for k in ((0, 0, True), (0, 3, False)))
    assert (a["gain"] > b["gain"]) is (direction == -1)
    assert 0 < abs(b["gain"] - a["gain"]) < 2*mass
    assert winner(options.values())["key"] == ((0, 0, True) if direction == -1 else (0, 3, False))


@pytest.mark.parametrize("stored", [False, True])
def test_row_enumeration_order_cannot_change_rational_gains(stored):
    x, fields = captured(stored=stored)
    forward = {r["key"]: r for r in candidates(x, fields, range(8))}
    reverse = {r["key"]: r for r in candidates(x, fields, reversed(range(8)))}
    assert {k: r["gain"] for k, r in forward.items()} == {k: r["gain"] for k, r in reverse.items()}
    assert winner(reversed(list(forward.values())))["key"] == winner(reverse.values())["key"]


def test_nonpositive_or_empty_candidates_do_not_split():
    x, fields = captured()
    assert winner(candidates(x, fields, ())) is None
    assert winner(candidates(x, [(Fraction(0), h) for _, h in fields], range(8))) is None
    assert len(exact_tree(x, fields, depth=0)) == 1
