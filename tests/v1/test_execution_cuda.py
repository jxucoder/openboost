"""Run only on real CUDA; no emulated arrays qualify storage ownership."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import numpy as np
import pytest

from openboost.execution import ExecutionContext

pytestmark = pytest.mark.gpu


def test_owned_upload_copy_export_and_lifetime():
    source = np.arange(8192 * 32, dtype=np.float32).reshape(8192, 32)
    expected = source.copy()
    with ExecutionContext() as context:
        original = context.upload(source)
        source[:] = -1
        copied = context.copy(original)
        context.release(original)
        with pytest.raises(ValueError, match="released"):
            context.export(original)
        exported = context.export(copied)
        np.testing.assert_array_equal(exported, expected)
        exported[:] = -2
        np.testing.assert_array_equal(context.export(copied), expected)
        assert context.metrics["upload_bytes"] == expected.nbytes
        assert context.metrics["device_copy_bytes"] == expected.nbytes
        assert context.metrics["export_bytes"] == 2 * expected.nbytes
        assert context.metrics["live_bytes"] == expected.nbytes
        assert context.metrics["peak_live_bytes"] == 2 * expected.nbytes
        assert context.metrics["peak_pool_bytes"] >= 2 * expected.nbytes
        print("storage_metrics=" + json.dumps(dict(context.metrics), sort_keys=True))
        with pytest.raises(TypeError):
            context.metrics["upload_bytes"] = 0
    assert context.metrics["live_bytes"] == 0
    with pytest.raises(RuntimeError, match="closed"):
        context.export(copied)
    context.close()


def test_foreign_forged_and_released_handles():
    with ExecutionContext() as a, ExecutionContext() as b:
        buffer = a.upload(np.array([1, 2], dtype=np.int32))
        for bad in (buffer, replace(buffer)):
            with pytest.raises(ValueError, match="foreign"):
                b.copy(bad)
        with pytest.raises(ValueError, match="forged"):
            a.export(replace(buffer))
        a.release(buffer)
        with pytest.raises(ValueError, match="released"):
            a.release(buffer)


def test_allocation_budget_and_context_recovery():
    with ExecutionContext(max_bytes=1024) as context:
        buffer = context.upload(np.zeros(128, dtype=np.float32))
        copy = context.copy(buffer)
        with pytest.raises(MemoryError, match="budget"):
            context.copy(buffer)
        assert context.metrics["live_bytes"] == 1024
        context.release(copy)
        np.testing.assert_array_equal(context.export(buffer), np.zeros(128))


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64, np.uint8, bool])
def test_dtype_and_noncontiguous_copy(dtype):
    value = np.arange(16).reshape(4, 4).astype(dtype)[:, ::2]
    with ExecutionContext() as context:
        buffer = context.upload(value)
        result = context.export(buffer)
        assert result.dtype == value.dtype
        np.testing.assert_array_equal(result, value)


def test_missing_values_and_zero_values_are_preserved():
    value = np.array([0, -0.0, np.nan, 2], dtype=np.float32)
    with ExecutionContext() as context:
        np.testing.assert_array_equal(
            context.export(context.upload(value)).view(np.uint32), value.view(np.uint32)
        )


def test_foreign_thread_and_current_stream_restored():
    import cupy as cp

    stream = cp.cuda.Stream(non_blocking=True)
    with stream, ExecutionContext() as context:
        buffer = context.upload(np.ones(3, dtype=np.float32))
        context.export(context.copy(buffer))
        assert cp.cuda.get_current_stream().ptr == stream.ptr
        with (
            ThreadPoolExecutor(max_workers=1) as pool,
            pytest.raises(RuntimeError, match="another thread"),
        ):
            pool.submit(context.export, buffer).result()


def test_device_input_and_unsupported_dtype_rejected():
    import cupy as cp

    with ExecutionContext() as context:
        with pytest.raises(ValueError, match="host data only"):
            context.upload(cp.zeros(2))
        for value in (np.array([], dtype=float), np.array(["x"]), np.ones(3, dtype=np.float16)):
            with pytest.raises(ValueError, match="host data required"):
                context.upload(value)
