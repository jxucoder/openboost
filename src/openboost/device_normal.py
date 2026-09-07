"""Experimental resident Normal geometry. Real-device validation is pending."""

from functools import partial

import numpy as np

from . import device_objectives as operations
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
        Normal.validate, prepare, partial(base, minimum_scale=floor), loss
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
