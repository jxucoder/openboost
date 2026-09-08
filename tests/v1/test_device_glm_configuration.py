"""Host configuration and explicit upload-domain checks without CUDA imports."""

from dataclasses import replace

import numpy as np
import pytest

from openboost import device_glm as glm

from .test_device_glm_reference import fixture


@pytest.mark.parametrize("factory", [glm.binary, glm.poisson])
@pytest.mark.parametrize("value", [True, -1, 0, 1e-60, np.inf, np.nan, 1e40, "1"])
def test_invalid_initialization_parameter(factory, value):
    keyword = "clip" if factory is glm.binary else "minimum_rate"
    with pytest.raises(ValueError):
        factory(**{keyword: value})


@pytest.mark.parametrize("clip", [0.5, 1, 1e-20])
def test_binary_probability_clip_is_representable(clip):
    with pytest.raises(ValueError):
        glm.binary(clip=clip)


@pytest.mark.parametrize("family", ["binary", "poisson"])
def test_upload_snapshot_and_explicit_missing_comparison(family):
    p = fixture(family)
    objective = getattr(glm, family)()
    objective.validate(p)
    assert objective.compare is None
    with pytest.raises(NotImplementedError, match="loss-change"):
        objective.loss_change(None, None, None, None)
    arrays = glm._host_arrays(p, family)
    assert len(arrays) == (3 if family == "poisson" else 2)
    for snapshot, original in zip(arrays, (p.target, p.offset, *p.structure.values()), strict=True):
        assert snapshot.dtype == np.float32 and not np.shares_memory(snapshot, original)
        np.testing.assert_array_equal(snapshot, original)
    arrays[0][0, 0] = 9
    assert p.target[0, 0] == 0


@pytest.mark.parametrize("change", ["count", "exposure", "offset"])
def test_unrepresentable_upload_rejected(change):
    p = fixture("poisson")
    if change == "count":
        p = replace(p, target=np.full((8, 1), 2**24 + 1))
    elif change == "exposure":
        p = replace(p, structure={"exposure": np.full((8, 1), 1e-60)})
    else:
        p = replace(p, offset=np.full((8, 1), 1e40))
    with pytest.raises(ValueError, match="float32"):
        glm._host_arrays(p, "poisson")


def test_exact_large_counts_allowed_and_wrong_family_rejected():
    p = replace(fixture("poisson"), target=np.full((8, 1), 2**24 + 2))
    np.testing.assert_array_equal(glm._host_arrays(p, "poisson")[0], p.target)
    with pytest.raises(ValueError, match="family"):
        glm._host_arrays(p, "unknown")
    for family, wrong in (("poisson", fixture("binary")), ("binary", p)):
        with pytest.raises(ValueError):
            glm._host_arrays(wrong, family)
