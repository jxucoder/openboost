"""Host-only multiclass upload/schema checks; never CUDA emulation."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import ClassSchema
from openboost import device_multiclass as multiclass
from openboost.device_runtime import DeviceRun

from .test_device_multiclass_reference import fixture


@pytest.mark.parametrize("width", [2, 3, 5])
def test_owned_upload_snapshot_and_explicit_comparison_dependency(width):
    p = fixture(width)
    objective = multiclass.objective()
    objective.validate(p)
    assert objective.compare is multiclass.compare
    missing_comparison = replace(objective, compare=None)
    with pytest.raises(NotImplementedError, match="loss-change"):
        missing_comparison.loss_change(None, None, None, None)
    with pytest.raises(NotImplementedError, match="loss-change"):
        DeviceRun(
            None,
            p,
            p,
            run_id="comparison",
            seed=7,
            objective=missing_comparison,
            comparison="objective",
        )
    assert objective.fields is objective.gradient is None
    arrays = multiclass._host_arrays(p)
    for snapshot, original in zip(arrays, (p.target, p.offset), strict=True):
        assert snapshot.dtype == np.float32 and not np.shares_memory(snapshot, original)
        np.testing.assert_array_equal(snapshot, original)
    arrays[0][0] = 9
    assert p.target[0, 0] == 0


@pytest.mark.parametrize("change", ["offset", "structure", "width", "classes"])
def test_invalid_host_upload_rejected(change):
    p = fixture()
    if change == "offset":
        p = replace(p, offset=np.full((9, 3), 1e40))
    elif change == "structure":
        p = replace(p, structure={"exposure": np.ones((9, 1))})
    elif change == "width":
        p = replace(p, raw_width=1, offset=np.zeros((9, 1)))
    else:
        p = replace(p, classes=None)
    with pytest.raises(ValueError):
        multiclass._host_arrays(p)


@pytest.mark.parametrize("channel", [True, -1, 3, 1.0, "1", None])
def test_class_channel_is_explicit_integer(channel):
    with pytest.raises(ValueError, match="class channel"):
        multiclass._channel(channel, 3)


def test_validation_class_schema_rejected_before_device_work():
    p = fixture()
    q = replace(fixture(validation=True), classes=ClassSchema(("a", "b", "c")))
    with pytest.raises(ValueError, match="class schemas differ"):
        DeviceRun(None, p, q, run_id="schema", seed=7, objective=multiclass.objective())


@pytest.mark.parametrize("configuration", [
    {"rounds": -1}, {"patience": 0}, {"min_delta": 1}, {"step": "unknown"},
    {"max_trials": 0}, {"max_trials": 7}, {"learning_rate": -1}, {"learning_rate": np.nan},
    {"max_depth": True}, {"reg_lambda": -1}, {"min_child_h": -1}, {"split_penalty": -1},
    {"learner": 3}, {"learner": lambda *_: None, "max_depth": 1},
])
def test_recipe_rejects_invalid_configuration_before_allocating(configuration):
    from openboost.device_recipes import multiclass as recipe
    with pytest.raises(ValueError):
        recipe(None, fixture(), fixture(validation=True), run_id="invalid", seed=7, **configuration)
