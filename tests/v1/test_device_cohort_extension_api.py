"""Source-only configuration checks; installed CUDA execution is separate."""

import numpy as np
import pytest

from .test_device_normal_reference import prepared_fixture
from .test_public_extensions import ROOT, load


@pytest.mark.parametrize(
    "information",
    [
        [],
        np.ones(6),
        np.ones((5, 2)),
        np.ones((6, 0)),
        [[np.nan]] * 6,
        [[np.inf]] * 6,
        [[-1]] * 6,
        [[1e40]] * 6,
    ],
)
def test_invalid_information_fails_before_device_allocation(information):
    module = load(ROOT / "cohort_splits/src/ob_cohort_splits/device.py", "cohort_device")
    train, _, binned, _ = prepared_fixture("d2")
    with pytest.raises(ValueError, match="cohort information"):
        module.DeviceCohortLearner(None, train, information, binning=binned.binning)
