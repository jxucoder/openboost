"""117 metadata validation; no fake execution context or CUDA simulation."""

import inspect

import pytest

from openboost import device_tree
from openboost.binning import Binning
from openboost.device import DeviceOperations, _vector_schema
from openboost.device_runtime import DeviceTerm


def test_vector_public_operations_and_reordered_metadata():
    for name in ("vector_scores", "vector_feasible", "vector_leaf"):
        assert callable(getattr(DeviceOperations, name))
    names = ("curvature:1", "cohort", "gradient:0", "curvature:0", "gradient:1")
    assert _vector_schema(
        names, ("training", "independent", "training", "training", "training")
    ) == ((2, 4), (3, 0))


@pytest.mark.parametrize(
    "names",
    [
        ("gradient", "curvature"),
        ("gradient:0",),
        ("curvature:0",),
        ("gradient:1", "curvature:1"),
        ("gradient:00", "curvature:0"),
        ("gradient:0", "curvature:0", "curvature:1"),
        ("gradient:0", "curvature:0", "gradient:2", "curvature:2"),
        ("gradient:0", "curvature:0", "gradient:x"),
        ("gradient:0", "curvature:0", "gradient:0"),
    ],
)
def test_incomplete_or_noncanonical_channels_rejected(names):
    with pytest.raises(ValueError):
        _vector_schema(names, ("training",) * len(names))


@pytest.mark.parametrize("role", ["unweighted", "independent"])
@pytest.mark.parametrize("column", [0, 1, 2, 3])
def test_every_channel_requires_training_role(role, column):
    roles = ["training"] * 4
    roles[column] = role
    with pytest.raises(ValueError, match="once-weighted"):
        _vector_schema(("gradient:0", "gradient:1", "curvature:0", "curvature:1"), roles)


def test_explicit_tree_width_and_scalar_runtime_boundary():
    assert "output_width" in inspect.signature(device_tree.depthwise).parameters
    assert "leaf_fields" in inspect.signature(device_tree.depthwise).parameters
    args = (Binning(("x",), ([],)), ((-1, -1, False, -1, -1),))
    assert device_tree.DeviceTree(*args).output_width == 1
    tree = device_tree.DeviceTree(*args, output_width=4)
    assert tree.output_width == 4 and tree.n_nodes == 1
    with pytest.raises(ValueError, match="width"):
        DeviceTerm(tree, [[1]])


@pytest.mark.parametrize("width", [0, -1, True, 1.5, None, "2"])
def test_invalid_tree_width(width):
    with pytest.raises(ValueError, match="width"):
        device_tree.DeviceTree(Binning(("x",), ([],)), (), output_width=width)
