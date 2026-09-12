"""Host prerequisites for the two run-13 controlled device recipe cases."""

import numpy as np
import pytest

from openboost import device_multiclass

from .test_device_multiclass_recipes_cuda import controlled


@pytest.mark.parametrize("initial,weights", [(2, None), (0, [3, 1, 1])])
def test_controlled_recipe_fixture_is_valid_before_device_allocation(initial, weights, monkeypatch):
    problem, binning = controlled(monkeypatch, initial=initial, weights=weights)
    device_multiclass.objective().validate(problem)
    assert problem.classes.values == ("a", "b", "c")
    np.testing.assert_array_equal(problem.target[:, 0], [0, 1, 2])
    np.testing.assert_array_equal(problem.weight, [1, 1, 1] if weights is None else weights)
    np.testing.assert_array_equal(problem.offset, np.zeros((3, 3)))
    assert binning.feature_names == problem.data.feature_names
