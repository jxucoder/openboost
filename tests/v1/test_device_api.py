"""CPU import and configuration checks do not imply real-device conformance."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from openboost import device_recipes
from openboost.data import ClassSchema
from openboost.device import DeviceOperations
from openboost.device_runtime import DeviceRun

from .test_device_round_reference import prepared_fixture


def test_device_module_import_does_not_require_cuda_packages():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.modules['cupy']=None; sys.modules['numba']=None; "
            "from openboost.device import DeviceOperations; "
            "from openboost import device_objectives, device_tree, device_runtime, device_recipes",
        ],
        check=True,
    )


@pytest.mark.parametrize("context", [None, "cuda:0", object()])
def test_explicit_context_required(context):
    with pytest.raises(ValueError, match="ExecutionContext"):
        DeviceOperations(context)


@pytest.mark.parametrize(
    "options",
    [
        {"rounds": -1},
        {"rounds": True},
        {"patience": 0},
        {"min_delta": 1},
        {"learning_rate": -1},
        {"learning_rate": np.inf},
        {"max_depth": -1},
        {"max_trials": 0},
        {"max_trials": 7},
        {"step": "unknown"},
        {"learner": object()},
        {"learner": lambda *args: None, "max_depth": 3},
    ],
)
def test_device_recipe_rejects_invalid_controls_before_execution(options):
    # No context or fake device is supplied: configuration fails before CUDA access.
    with pytest.raises(ValueError):
        device_recipes.squared(None, None, None, run_id="invalid", seed=0, **options)


@pytest.mark.parametrize("kind", ["vector", "class", "structure"])
def test_device_run_rejects_unsupported_problem_before_execution(kind):
    train, validation, _, _ = prepared_fixture("d2")
    if kind == "vector":
        train = replace(
            train,
            target=np.tile(train.target, (1, 2)),
            offset=np.tile(train.offset, (1, 2)),
            raw_width=2,
        )
    elif kind == "class":
        train = replace(
            train, target=np.zeros_like(train.target), classes=ClassSchema(("no", "yes"))
        )
    else:
        train = replace(train, structure={"groups": np.arange(len(train.target))[:, None]})
    # Public run validation rejects unsupported semantics before touching CUDA storage.
    with pytest.raises(ValueError, match="scalar"):
        DeviceRun(None, train, validation, run_id="invalid", seed=0)
