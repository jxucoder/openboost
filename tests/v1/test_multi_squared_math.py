"""118 production scalar expressions versus independent rational mathematics."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from openboost import device_multi_squared as objective
from openboost._comparison_math import cpu_add, cpu_div, cpu_mul
from openboost._multi_squared_math import cpu_channel_change
from openboost.device_objectives import _projection

from .reference import multi_squared as ref
from .test_multi_squared_reference import prepared


def comparison(arrays):
    old, new, target, offset, weight = arrays
    total, mass = (0.0, 0.0), (0.0, 0.0)
    for r in range(len(old)):
        row = (0.0, 0.0)
        for k in range(old.shape[1]):
            lo, hi, code = cpu_channel_change(*(float(a[r, k]) for a in (old, new, target, offset)))
            assert code == 0
            row = cpu_add(row, (lo, hi))
        w = (float(weight[r]),) * 2
        total, mass = cpu_add(total, cpu_mul(row, w)), cpu_add(mass, w)
    return objective._result(*cpu_div(total, mass), 0, int(np.array_equal(old, new)))


@pytest.mark.parametrize("case", ref.CASES, ids=lambda c: c["id"])
@pytest.mark.parametrize("reverse", [False, True])
def test_directed_polynomial_encloses_exact_stored_change(case, reverse):
    old, new, *rest = case["arrays"]
    arrays = (new, old, *rest) if reverse else (old, new, *rest)
    exact, result = ref.change(*arrays), comparison(arrays)
    assert Fraction(result.lower) <= exact <= Fraction(result.upper)
    if exact:
        assert result.status == ("improvement" if exact < 0 else "worsening")
    else:
        assert not result.improves()
    assert result.unchanged == np.array_equal(old, new)


@pytest.mark.parametrize("width", [1, 2, 4])
def test_objective_surface_and_owned_host_inputs(width):
    problem, _, _ = prepared(width)
    arrays = objective._host_arrays(problem)
    assert all(
        a.dtype == np.float32 and not np.shares_memory(a, p)
        for a, p in zip(arrays, (problem.target, problem.offset), strict=True)
    )
    obj = objective.objective()
    assert all(callable(f) for f in (obj.prepare, obj.base, obj.loss, obj.compare))
    with pytest.raises(ValueError, match="float32"):
        objective._host_arrays(replace(problem, target=problem.target + 1e100))
    with pytest.raises(ValueError, match="multi-output"):
        objective._host_arrays(replace(problem, structure={"exposure": np.ones((8, 1))}))


@pytest.mark.parametrize(
    "value",
    [
        [],
        [1, 2],
        [[0], [0]],
        [[1]],
        [[1, np.inf], [1, 0]],
        [[1e100], [0]],
        [[1e-100], [0]],
        np.array([[1j], [1]]),
    ],
)
def test_projection_rejects_unusable_or_mismatched_columns(value):
    with pytest.raises(ValueError):
        _projection(value, 2)


def test_projection_metadata_owned_and_immutable():
    source = np.array([[1.0, -0.5], [0, 1]])
    actual = _projection(source, 2)
    source[:] = 0
    np.testing.assert_array_equal(actual, [[1, -0.5], [0, 1]])
    with pytest.raises(ValueError):
        actual.setflags(write=True)
    assert objective._result(0, 0, 2, 0).reason == "arithmetic_range"
    with pytest.raises(RuntimeError):
        objective._result(0, 0, 1, 0)
