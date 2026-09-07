"""Freeze numerical support and tolerances without executing a simulated GPU."""

import numpy as np
import pytest

from .reference.normal_precision import DOMAIN_CASES, stored_geometry


@pytest.mark.parametrize(
    "name,raw,target,offset,valid", DOMAIN_CASES, ids=[c[0] for c in DOMAIN_CASES]
)
def test_declared_float32_domain(name, raw, target, offset, valid):
    if valid:
        loss, gradient, fisher = stored_geometry([raw], [target], [offset], [1])
        assert np.isfinite(loss) and gradient.dtype == fisher.dtype == np.dtype("float32")
        assert np.all(fisher > 0)
    else:
        with pytest.raises((ValueError, OverflowError)):
            stored_geometry([raw], [target], [offset], [1])


def test_float32_invalid_trials_recover_without_clipping():
    # The same 090-A direction has a narrower representable trial domain.
    valid = []
    for j in range(6):
        alpha = 0.2 * 0.5**j
        try:
            value, _, _ = stored_geometry([[alpha * 100, alpha * 5000]], [100], [[0, 0]], [1])
        except (ValueError, OverflowError):
            valid.append(False)
        else:
            valid.append(True)
            assert value < 5000 + 0.5 * np.log(2 * np.pi)
    assert valid == [False, False, False, False, False, True]


def test_invalid_zero_weight_row_is_not_hidden():
    with pytest.raises(ValueError):
        stored_geometry([[0, 0], [0, -50]], [0, 0], [[0, 0], [0, 0]], [1, 0])
