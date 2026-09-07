"""Vector layout reuse must not share mutable indices or change algorithm results."""

import numpy as np
import pytest

from openboost import RunContext, ops
from openboost.recipes import multi_squared
from openboost.tree import best_first, depthwise, symmetric
from tests.v1.test_public_multioutput import fixture


def uncached(names):
    width = sum(n.startswith("gradient:") for n in names)
    if width == 0:
        raise ValueError("vector Newton fields required")
    return (
        [names.index(f"gradient:{k}") for k in range(width)],
        [names.index(f"curvature:{k}") for k in range(width)],
    )


def test_repeated_layout_is_resolved_once_and_cannot_be_poisoned():
    class Field(str):
        calls = 0

        def startswith(self, prefix):
            Field.calls += 1
            return super().startswith(prefix)

    names = tuple(
        map(Field, ("gradient:0", "curvature:0", "gradient:1", "curvature:1", "cache-probe"))
    )
    first = ops._vector_indices(names)
    count = Field.calls
    first[0][0] = 99
    assert ops._vector_indices(list(names)) == ([0, 2], [1, 3])
    assert Field.calls == count


@pytest.mark.parametrize("names", [(), ("gradient:1", "curvature:1"), ("gradient:0",)])
def test_malformed_layout_remains_rejected(names):
    with pytest.raises(ValueError):
        ops._vector_indices(names)


@pytest.mark.parametrize("policy", [depthwise, best_first, symmetric])
@pytest.mark.parametrize("projection", [None, np.array([[1.0], [0.0]])])
def test_cached_and_original_layouts_have_exact_models_and_predictions(
    monkeypatch, tmp_path, policy, projection
):
    p = fixture()
    options = dict(
        context=RunContext("layout-parity", 1),
        rounds=3,
        bins=4,
        grower=policy,
        projection=projection,
    )
    cached = multi_squared(p, p, **options).state.model
    monkeypatch.setattr(ops, "_vector_indices", uncached)
    original = multi_squared(p, p, **options).state.model
    cached.save(tmp_path / "cached.json")
    original.save(tmp_path / "original.json")
    assert (tmp_path / "cached.json").read_bytes() == (tmp_path / "original.json").read_bytes()
    np.testing.assert_array_equal(cached.predict(p.data), original.predict(p.data))


@pytest.mark.parametrize(
    "total,regularizer",
    [
        ([1.0, 0.0], 0.0),
        ([1.0, -1.0], 1.0),
        ([float("nan"), 1.0], 1.0),
        ([1.0, 1.0], -1.0),
        ([1e308, 1e-308], 0.0),
    ],
)
def test_leaf_failure_behavior_is_unchanged(monkeypatch, total, regularizer):
    names = ("gradient:0", "curvature:0")
    with pytest.raises(ValueError) as cached:
        ops.vector_leaf(np.array(total), names, reg_lambda=regularizer)
    monkeypatch.setattr(ops, "_vector_indices", uncached)
    with pytest.raises(ValueError) as original:
        ops.vector_leaf(np.array(total), names, reg_lambda=regularizer)
    assert str(cached.value) == str(original.value)
