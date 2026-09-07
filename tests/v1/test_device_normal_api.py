"""Normal device configuration/import checks require no CUDA execution."""

import subprocess
import sys

import numpy as np
import pytest

from openboost import device_normal, device_recipes
from openboost.binning import Binning
from openboost.device_runtime import DeviceTerm
from openboost.device_tree import DeviceTree


@pytest.mark.parametrize("floor", [0, -1, True, None, "1", np.nan, np.inf, 1e100, 1e-100])
def test_invalid_initial_scale_floor(floor):
    with pytest.raises(ValueError):
        device_normal.objective(minimum_scale=floor)


def test_import_and_configuration_without_cuda_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import sys; sys.modules['cupy']=None; sys.modules['numba']=None",
                    "from openboost import device_normal",
                    "operations = device_normal.objective()",
                    "assert callable(operations.loss) and callable(operations.prepare)",
                ]
            ),
        ],
        check=True,
    )


def metadata_tree():
    # Host metadata only; attempting execution requires a registered real device tree.
    return DeviceTree(Binning(("x",), (np.array([]),)), ((-1, -1, False, -1, -1),))


def test_term_owns_immutable_mapping_metadata():
    source = np.array([[1, -0.5]], np.float32)
    term = DeviceTerm(metadata_tree(), source)
    source[:] = 0
    np.testing.assert_array_equal(term.mapping, [[1, -0.5]])
    with pytest.raises(ValueError):
        term.mapping.setflags(write=True)


@pytest.mark.parametrize(
    "mapping", [[], [1, 2], [[1], [2]], [[np.inf]], [[1e100]], [[None]], [[1j]]]
)
def test_invalid_mapping_metadata(mapping):
    with pytest.raises(ValueError):
        DeviceTerm(metadata_tree(), mapping)


@pytest.mark.parametrize(
    "options",
    [
        {"update": "unknown"},
        {"mode": "full"},
        {"mode": "ordinary", "damping": 1},
        {"damping": -1},
        {"minimum_scale": 0},
        {"rounds": -1},
        {"rounds": True},
        {"max_trials": 7},
        {"step": "other"},
        {"max_depth": -1},
        {"patience": 0},
        {"learning_rate": np.nan},
        {"learner": object()},
        {"learner": lambda *args: None, "max_depth": 3},
    ],
)
def test_normal_configuration_fails_before_cuda_execution(options):
    with pytest.raises(ValueError):
        device_recipes.normal(None, None, None, run_id="invalid", seed=7, **options)
