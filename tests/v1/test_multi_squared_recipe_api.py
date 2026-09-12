"""118 general mapping ownership and early recipe configuration controls."""

import numpy as np
import pytest

from openboost import device_recipes as recipes
from openboost.binning import Binning
from openboost.device_runtime import DeviceTerm
from openboost.device_tree import DeviceTree

from .test_multi_squared_reference import prepared


@pytest.mark.parametrize("width,raw_width", [(1, 3), (2, 1), (2, 4), (4, 2)])
def test_general_map_owns_immutable_metadata(width, raw_width):
    tree = DeviceTree(Binning(("x",), ([],)), ((-1, -1, False, -1, -1),), width)
    source = np.ones((width, raw_width), np.float32)
    term = DeviceTerm(tree, source)
    source[:] = 0
    np.testing.assert_array_equal(term.mapping, np.ones((width, raw_width)))
    with pytest.raises(ValueError):
        term.mapping.setflags(write=True)
    with pytest.raises(ValueError, match="width"):
        DeviceTerm(tree, np.ones((width + 1, raw_width)))
    with pytest.raises(ValueError):
        DeviceTerm(tree, np.ones((width, raw_width), complex))


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "bad"},
        {"mode": "independent", "projection": [[1], [0]]},
        {"projection": [[0], [0]]},
        {"max_depth": True},
        {"reg_lambda": -1},
        {"min_child_h": np.inf},
        {"split_penalty": -1},
        {"rounds": -1},
        {"learning_rate": np.inf},
        {"step": "bad"},
        {"max_trials": 0},
        {"patience": 0},
        {"learner": object()},
        {"learner": lambda *a: None, "max_depth": 3},
    ],
)
def test_bad_recipe_configuration_fails_before_device_access(options):
    train, validation, _ = prepared()
    with pytest.raises(ValueError):
        recipes.multi_squared(None, train, validation, run_id="bad", seed=7, **options)
