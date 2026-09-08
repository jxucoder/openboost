"""105 real-device boolean validation, tail lanes and public failure recovery."""

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

pytestmark = pytest.mark.gpu
SHAPES = [(0, 1), (1, 2), (127, 17), (128, 2), (129, 1), (257, 17), (8193, 2), (100003, 2)]


@pytest.mark.parametrize("rows,columns", SHAPES)
@pytest.mark.parametrize("nonnegative", [False, True])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_all_column_flags_match_independent_any(rows, columns, nonnegative, dtype):
    values = np.ones((rows, columns), dtype)
    if rows:
        values[-1] = np.resize(np.array([np.nan, np.inf, -np.inf, -1, -0.0, 1], dtype), columns)
    expected = np.any(~np.isfinite(values) | ((values < 0) if nonnegative else False), axis=0)
    with ExecutionContext(max_bytes=32 * 1024**2) as context:
        ops = DeviceOperations(context)
        buffer = context.upload(values) if rows else context._empty(values.shape, dtype)
        before = context.metrics["live_bytes"]
        if expected.any():
            with pytest.raises(ValueError, match="finite"):
                ops._validate(context._array(buffer), nonnegative=nonnegative)
        else:
            ops._validate(context._array(buffer), nonnegative=nonnegative)
        assert context.metrics["live_bytes"] == before
        # Read the actual per-column flags separately from the public error check.
        flags = context.upload(np.full(columns, 99, np.int32))
        ops._launch(
            "validate_fields",
            columns * 128,
            context._array(buffer),
            nonnegative,
            context._array(flags),
        )
        np.testing.assert_array_equal(context.export(flags), expected.astype(np.int32))
        np.testing.assert_array_equal(context.export(buffer), values)
        context.release(flags)
        context.release(buffer)
        assert context.metrics["live_bytes"] == 0


def prepared(context, rows=257):
    ids = np.arange(rows)
    data = NumericData(ids[:, None], ids, ("x",))
    weights = np.ones(rows)
    weights[-1] = 0
    problem = Problem(data, np.zeros((rows, 1)), ids, weight=weights)
    ops = DeviceOperations(context)
    resident = ops.prepare(Binning.fit(data, bins=4).transform(data), problem)
    return ops, resident


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_public_fields_reject_invalid_tail_even_with_zero_weight(bad):
    with ExecutionContext() as context:
        ops, data = prepared(context)
        values = np.ones((257, 2), np.float32)
        values[-1, -1] = bad
        buffer = context.upload(values)
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="finite"):
            ops.fields(data, buffer, names=("g", "h"), roles=("unweighted", "unweighted"))
        assert context.metrics["live_bytes"] == before
        np.testing.assert_array_equal(context.export(buffer), values)


def test_public_independent_nonnegative_tail_and_negative_zero():
    with ExecutionContext() as context:
        ops, data = prepared(context)
        fields = ops.fields(
            data,
            context.upload(np.ones((257, 2), np.float32)),
            names=("g", "h"),
            roles=("unweighted", "unweighted"),
        )
        column = np.ones(257, np.float32)
        column[-1] = -1
        negative = context.upload(column)
        before = context.metrics["live_bytes"]
        with pytest.raises(ValueError, match="nonnegative"):
            ops.add_independent(fields, "cohort", negative, nonnegative=True)
        assert context.metrics["live_bytes"] == before
        column[-1] = -0.0
        valid = ops.add_independent(fields, "cohort", context.upload(column), nonnegative=True)
        np.testing.assert_array_equal(context.export(valid.values)[:, -1], column)


@pytest.mark.parametrize("failure", ["allocate", "before_launch", "after_launch"])
def test_validation_failure_preserves_borrowed_buffer_and_recovers(monkeypatch, failure):
    with ExecutionContext() as context:
        ops, data = prepared(context)
        values = np.ones((257, 2), np.float32)
        buffer = context.upload(values)
        before = context.metrics["live_bytes"]
        with monkeypatch.context() as patch:
            if failure == "allocate":

                def fail(*args, **kwargs):
                    raise MemoryError("injected flag allocation")

                patch.setattr(context, "_empty", fail)
            else:
                original = ops._launch

                def fail(name, *args):
                    if failure == "after_launch":
                        original(name, *args)
                    raise RuntimeError("injected validation launch")

                patch.setattr(ops, "_launch", fail)
            with pytest.raises((MemoryError, RuntimeError), match="injected"):
                ops.fields(data, buffer, names=("g", "h"), roles=("unweighted", "unweighted"))
        assert context.metrics["live_bytes"] == before
        np.testing.assert_array_equal(context.export(buffer), values)
        recovered = ops.fields(data, buffer, names=("g", "h"), roles=("unweighted", "unweighted"))
        np.testing.assert_array_equal(context.export(recovered.values), values)
