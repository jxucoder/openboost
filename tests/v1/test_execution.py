"""CPU-side validation never treats an unavailable device as a CPU fallback."""

import pytest

from openboost.execution import ExecutionContext


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:-1", "cuda:x", None])
def test_invalid_explicit_device(device):
    with pytest.raises(ValueError, match="cuda"):
        ExecutionContext(device)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_allocation_limit_precedes_device_import(limit):
    with pytest.raises(ValueError, match="max_bytes"):
        ExecutionContext("cuda:0", max_bytes=limit)


def test_missing_cupy_cannot_fall_back(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(RuntimeError, match="fallback is unavailable"):
        ExecutionContext()
