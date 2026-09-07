"""Scratch scoring must match the original immutable-leaf score exactly."""

from dataclasses import replace
from functools import partial

import numpy as np
import pytest

from openboost import RunContext, ops, recipes
from openboost.tree import best_first, depthwise, symmetric
from tests.v1.test_public_multioutput import fixture


def original(candidate, *, reg_lambda=1.0, split_penalty=0.0):
    g, _ = ops._vector_indices(candidate.names)

    def node(total):
        with np.errstate(over="raise", invalid="raise"):
            return -0.5 * np.dot(
                total[g], ops.vector_leaf(total, candidate.names, reg_lambda=reg_lambda)
            )

    value = (
        node(candidate.left)
        + node(candidate.right)
        - node(candidate.parent)
        - ops._nonnegative(split_penalty)
    )
    if not np.isfinite(value):
        raise ValueError("nonfinite vector split score")
    return float(value)


def candidate(width=2):
    rng = np.random.default_rng(width)
    names = tuple(n for k in range(width) for n in (f"gradient:{k}", f"curvature:{k}"))
    left = np.column_stack((rng.normal(size=width), rng.uniform(0.1, 3, width))).ravel()
    right = np.column_stack((rng.normal(size=width), rng.uniform(0.1, 3, width))).ravel()
    return ops.Candidate(
        0,
        1,
        False,
        names,
        ("weighted",) * len(names),
        left,
        right,
        left + right,
        2,
        3,
        "data",
        "rows",
    )


def test_scoring_does_not_construct_owned_leaf_artifacts(monkeypatch):
    c = candidate()
    expected = original(c)

    def forbidden(*args, **kwargs):
        raise AssertionError("scoring constructed an immutable leaf")

    monkeypatch.setattr(ops, "vector_leaf", forbidden)
    assert ops.vector_score(c) == expected


@pytest.mark.parametrize("width", [1, 2, 8])
@pytest.mark.parametrize("regularizer", [0.0, 1.0, 10.0])
def test_scores_ties_and_near_ties_are_exact(width, regularizer):
    c = candidate(width)
    other = replace(c, threshold=0, left=c.left.copy())
    other.left[0] = np.nextafter(other.left[0], np.inf)
    for option in (c, other):
        assert ops.vector_score(option, reg_lambda=regularizer) == original(
            option, reg_lambda=regularizer
        )
    for options in ([c, replace(c, threshold=0)], [c, other]):
        assert ops.choose(
            options,
            scoring=partial(ops.vector_score, reg_lambda=regularizer),
            legality=ops.vector_feasible,
        ) is ops.choose(
            options, scoring=partial(original, reg_lambda=regularizer), legality=ops.vector_feasible
        )


@pytest.mark.parametrize(
    "fault", ["negative_h", "zero_h", "nan_g", "infinite_h", "overflow", "negative_reg"]
)
def test_invalid_scores_remain_rejected(fault):
    c = candidate(1)
    reg = 0.0
    if fault == "negative_h":
        c.left[1] = -1
    elif fault == "zero_h":
        c.left[1] = 0
    elif fault == "nan_g":
        c.left[0] = np.nan
    elif fault == "infinite_h":
        c.left[1] = np.inf
    elif fault == "overflow":
        c.left[:] = [1e308, 1e-308]
    else:
        reg = -1
    with pytest.raises((ValueError, FloatingPointError)) as old:
        original(c, reg_lambda=reg)
    with pytest.raises(type(old.value)):
        ops.vector_score(c, reg_lambda=reg)


@pytest.mark.parametrize("grower", [depthwise, best_first, symmetric])
def test_recipe_model_bytes_match_original_score(monkeypatch, tmp_path, grower):
    p = fixture()
    options = dict(context=RunContext("scratch", 1), rounds=3, grower=grower, bins=4)
    actual = recipes.multi_squared(p, p, **options).state.model
    monkeypatch.setattr(recipes, "vector_score", original)
    expected = recipes.multi_squared(p, p, **options).state.model
    actual.save(tmp_path / "actual.json")
    expected.save(tmp_path / "expected.json")
    assert (tmp_path / "actual.json").read_bytes() == (tmp_path / "expected.json").read_bytes()
    np.testing.assert_array_equal(actual.predict(p.data), expected.predict(p.data))
