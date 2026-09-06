"""Counterexamples for real-data worker packet boundaries."""

import copy

import numpy as np
import pytest
from benchmarks.v1.freeze_preprocessing import prepare
from benchmarks.v1.worker_data import bind


def fixture():
    data = dict(
        x=np.arange(18, dtype=float).reshape(9, 2),
        y=np.column_stack([np.arange(9), np.full(9, 7)]),
        group=np.repeat(np.arange(3), 3),
    )
    parts = dict(train=np.arange(3), validation=np.arange(3, 6), test=np.arange(6, 9))
    frozen = prepare("parkinsons", data, [parts])[0]
    return data, parts, frozen


def test_packet_boundary_and_original_target_units():
    data, parts, frozen = fixture()
    packets, meta = bind("A6", data, parts, frozen)
    assert not any("test" in key for key in packets["worker-input"])
    assert set(packets["test-features"]) == {"row_ids", "x"}
    np.testing.assert_array_equal(packets["worker-input"]["y_train"], data["y"][:3])
    assert meta["target_scale"]["mean"] == [1, 7]
    assert meta["target_scale"]["constant"] == [False, True]


@pytest.mark.parametrize("change", ["order", "encoder", "overlap", "target", "group"])
def test_corrupted_binding_is_rejected(change):
    data, parts, frozen = fixture()
    if change == "order":
        parts["validation"] = parts["validation"][::-1]
    elif change == "encoder":
        frozen["encoder"]["median"][0] += 1
    elif change == "overlap":
        parts["test"][0] = 0
    elif change == "target":
        data["y"][0, 0] += 10
    else:
        data["group"][3] = 0
    with pytest.raises(ValueError):
        bind("A6", data, parts, frozen)


def test_concrete_age_feature_and_support_stay_aligned():
    data, parts, _ = fixture()
    data["y"] = data["y"][:, 0]
    data["structure"] = np.array([1, 2, 3, 4, 5, 6, 0.5, 8, 9])
    frozen = prepare("concrete", data, [parts])[0]
    original = copy.deepcopy(data)
    packets, meta = bind("A12", data, parts, frozen)
    np.testing.assert_array_equal(packets["test-features"]["x"][:, -1], [0.5, 8, 9])
    assert meta["age_train_range"] == [1, 3]
    np.testing.assert_array_equal(packets["test-structure"]["row_ids"], parts["test"])
    np.testing.assert_array_equal(data["x"], original["x"])
    assert packets["worker-input"]["x_train"].shape[1] == 5
