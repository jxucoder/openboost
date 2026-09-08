"""Experimental resident Normal geometry. Real-device validation is pending."""

from functools import partial

import numpy as np

from . import device_objectives as operations
from .comparison import _normal_result
from .device import _atomic, _parameter, _workspace
from .objectives import Normal

prepare = partial(operations.prepare, validate=Normal.validate)


def _floor(value):
    value = _parameter(value)
    if value <= 0:
        raise ValueError("positive finite float32 minimum scale required")
    return value


def objective(*, minimum_scale=1e-6):
    """Explicit Normal dependencies for the shared device transaction runtime."""
    floor = _floor(minimum_scale)
    return operations.ObjectiveOperations(
        Normal.validate, prepare, partial(base, minimum_scale=floor), loss, compare=compare
    )


@_atomic
def base(ops, problem, *, minimum_scale=1e-6):
    """Resident weighted offset-aware initialization; the floor is initialization only."""
    minimum_scale = _floor(minimum_scale)
    target, offset = operations._arrays(ops, problem, widths=(1, 2))
    with _workspace(ops) as retained:
        output = ops.execution._empty((2,), np.float32)
        ops._launch(
            "normal_base",
            1,
            target,
            offset,
            ops.execution._array(problem.data.weight),
            minimum_scale,
            ops.execution._array(output),
        )
        ops._validate(ops.execution._array(output).reshape(1, 2))
        raw = operations.broadcast(ops, output, problem.data.n_rows)
        geometry(
            ops, problem, raw
        )  # Validate representability on every row, including weight zero.
        retained.add(output)
        return output


@_atomic
def geometry(ops, problem, raw):
    """Owned unweighted gradient/Fisher [N,2]; offsets applied exactly once."""
    target, offset = operations._arrays(ops, problem, widths=(1, 2))
    values = ops._float(raw, (problem.data.n_rows, 2))
    ops._validate(values)
    g, h = (ops.execution._empty(values.shape, np.float32) for _ in range(2))
    ops._launch(
        "normal_geometry",
        problem.data.n_rows,
        target,
        offset,
        values,
        ops.execution._array(g),
        ops.execution._array(h),
    )
    ops._validate(ops.execution._array(g), message="Normal geometry outside float32 support")
    ops._validate(ops.execution._array(h), message="Normal geometry outside float32 support")
    return g, h


@_atomic
def loss(ops, problem, raw):
    """Weighted mean Normal NLL; only one float64 metric is exported."""
    target, offset = operations._arrays(ops, problem, widths=(1, 2))
    values = ops._float(raw, (problem.data.n_rows, 2))
    ops._validate(values)
    output = ops.execution._empty((1,), np.float64)
    ops._launch(
        "normal_loss",
        1,
        target,
        offset,
        ops.execution._array(problem.data.weight),
        values,
        ops.execution._array(output),
    )
    result = float(ops.execution.export(output)[0])
    ops.execution._counts.setdefault("metric_export_bytes", 0)
    ops.execution._counts["metric_export_bytes"] += output.nbytes
    ops.execution.release(output)
    if not np.isfinite(result):
        raise ValueError("Normal geometry outside float32 support")
    return result


@_atomic
def compare(ops, problem, before, after):
    """Resident stored-input NLL change with scalar bounds; CUDA validation pending.

    Domain/shape checks include every row. Both inputs remain owned by the caller.
    No raw/gradient transfer or CPU comparison occurs. Only a 32-byte summary and
    existing validation flags leave the device; all local scratch is released.
    """
    target, offset = operations._arrays(ops, problem, widths=(1, 2))
    shape = (problem.data.n_rows, 2)
    old, new = ops._float(before, shape), ops._float(after, shape)
    ops._validate(old)
    ops._validate(new)
    context = ops.execution
    with _workspace(ops):
        rows = context._empty((shape[0], 4), np.float64)
        output = context._empty((4,), np.float64)
        context._counts.setdefault("comparison_calls", 0)
        context._counts["comparison_calls"] += 1
        ops._launch("normal_compare_rows", shape[0], target, offset, old, new, context._array(rows))
        ops._launch(
            "normal_compare_reduce",
            1,
            context._array(rows),
            context._array(problem.data.weight),
            context._array(output),
        )
        summary = context.export(output)
        context._counts.setdefault("comparison_export_bytes", 0)
        context._counts["comparison_export_bytes"] += output.nbytes
        lower, upper, code, unchanged = (float(value) for value in summary)
        if code == 3:
            raise ValueError("Normal geometry outside float32 support")
        if code not in (0, 1, 2) or unchanged not in (0, 1):
            raise RuntimeError("invalid device comparison summary")
        return _normal_result(lower, upper, int(code), bool(unchanged))
