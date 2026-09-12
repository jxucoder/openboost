"""Host AFT preparation contracts must fail before device allocation."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import device_aft as aft

from .test_device_aft_reference import fixture


@pytest.mark.parametrize("sigma", [True, False, 0, -1, np.nan, np.inf, "1", [1], 1e-300, 1e300])
def test_invalid_fixed_scale(sigma):
    with pytest.raises(ValueError, match="scale"):
        aft.objective(sigma)


@pytest.mark.parametrize("sigma", [0.5, 0.7, 1, 2, 1e-20, 1e20])
def test_prepared_metadata_preserves_scale_and_censoring(sigma):
    p = fixture()
    objective = aft.objective(sigma)
    objective.validate(p)
    assert objective.prepare.keywords["sigma"] == float(sigma)
    assert objective.compare.func is aft.compare
    assert objective.compare.keywords["sigma"] == float(sigma)
    lower, offset, event = aft._host_arrays(p)
    assert lower.dtype == offset.dtype == np.float32 and event.dtype == bool
    np.testing.assert_array_equal(lower, p.target[:, :1])
    np.testing.assert_array_equal(offset, p.offset)
    np.testing.assert_array_equal(event, p.target[:, 0] == p.target[:, 1])
    assert not np.shares_memory(lower, p.target)
    lower[0, 0] = 99
    assert p.target[0, 0] == 1


@pytest.mark.parametrize("kind", ["tiny_time", "large_time", "large_offset", "numeric_target", "structure"])
def test_unrepresentable_or_wrong_target_rejected(kind):
    p = fixture()
    if kind in ("tiny_time", "large_time"):
        lower = np.full(8, 1e-60 if kind == "tiny_time" else 1e40)
        p = replace(p, target=np.column_stack((lower, np.full(8, np.inf))))
    elif kind == "large_offset":
        p = replace(p, offset=np.full((8, 1), 1e40))
    elif kind == "numeric_target":
        p = replace(p, target=p.target[:, :1], target_kind="numeric")
    else:
        p = replace(p, structure={"exposure": np.ones((8, 1))})
    with pytest.raises(ValueError):
        aft._host_arrays(p)
