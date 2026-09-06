import numpy as np
import pytest
from bounded_leaves import BoundedNewton

from openboost.experimental import ExecutionContext, TrainerConfig


def test_bounded_reference_and_zero_nodes():
    ctx = ExecutionContext("cpu", np, np.random.default_rng(7), 0, "mu")
    G = np.array([0, 8, -6, 0], np.float32)
    H = np.array([0, 1, 2, 3], np.float32)
    result = BoundedNewton(0.5).values(G, H, config=TrainerConfig(), context=ctx)
    np.testing.assert_array_equal(result, [0, -0.5, 0.5, 0])
    assert result.dtype == np.float32
    np.testing.assert_array_equal(G, [0, 8, -6, 0])
    for value in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            BoundedNewton(value)


def test_declared_devices():
    assert BoundedNewton.supported_devices == frozenset({"cpu", "cuda"})
