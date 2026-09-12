"""Experimental resident multi-output squared geometry and loss-change operations."""

from dataclasses import dataclass

import numpy as np

from . import device_objectives as operations
from .comparison import LossChange
from .device import DeviceData, _atomic, _workspace
from .objectives import MultiSquared


@dataclass(frozen=True, eq=False)
class MultiSquaredProblem(operations.DeviceProblem):
    """Family-bound targets and offsets with matching K columns."""


def objective():
    return operations.ObjectiveOperations(
        MultiSquared.validate, prepare, base, loss, compare=compare
    )


def _host_arrays(problem):
    MultiSquared.validate(problem)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            arrays = tuple(a.astype(np.float32) for a in (problem.target, problem.offset))
        except FloatingPointError as error:
            raise ValueError("finite float32 multi-output inputs required") from error
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("finite float32 multi-output inputs required")
    return arrays


@_atomic
def prepare(ops, data, problem):
    arrays = _host_arrays(problem)
    ops._get(data, DeviceData)
    if data.problem_identity != problem.identity:
        raise ValueError("prepared problem identity differs")
    handles = tuple(ops.execution.upload(a) for a in arrays)
    return ops._record(
        MultiSquaredProblem(data, problem.raw_width, problem.raw_width), handles, handles
    )


def _arrays(ops, problem):
    ops._get(problem, operations.DeviceProblem)
    if type(problem) is not MultiSquaredProblem:
        raise ValueError("prepared objective family differs")
    if problem.raw_width < 1 or problem.target_width != problem.raw_width:
        raise ValueError("matching multi-output target/raw widths required")
    return tuple(ops.execution._array(h) for h in ops._records[problem][0])


@_atomic
def base(ops, problem):
    """Weighted means of target minus offset, accumulated in float64 and stored float32."""
    target, offset = _arrays(ops, problem)
    with _workspace(ops) as retained:
        output = ops.execution._empty((problem.raw_width,), np.float32)
        ops._launch(
            "multi_squared_base",
            problem.raw_width,
            target,
            offset,
            ops.execution._array(problem.data.weight),
            ops.execution._array(output),
        )
        ops._validate(ops.execution._array(output).reshape(1, -1))
        geometry(ops, problem, operations.broadcast(ops, output, problem.data.n_rows))
        retained.add(output)
        return output


@_atomic
def geometry(ops, problem, raw):
    """Independent owned unweighted [N,K] gradient and unit diagonal curvature."""
    target, offset = _arrays(ops, problem)
    values = ops._float(raw, (problem.data.n_rows, problem.raw_width))
    ops._validate(values)
    g, h = (ops.execution._empty(values.shape, np.float32) for _ in range(2))
    ops._launch(
        "multi_squared_geometry",
        problem.data.n_rows,
        target,
        offset,
        values,
        ops.execution._array(g),
        ops.execution._array(h),
    )
    ops._validate(ops.execution._array(g), message="multi-output geometry outside float32 support")
    return g, h


@_atomic
def loss(ops, problem, raw):
    """Weighted mean of summed output half-squares; export one float64 reporting metric."""
    target, offset = _arrays(ops, problem)
    values = ops._float(raw, (problem.data.n_rows, problem.raw_width))
    ops._validate(values)
    with _workspace(ops):
        output = ops.execution._empty((1,), np.float64)
        ops._launch(
            "multi_squared_loss",
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
        if not np.isfinite(result):
            raise ValueError("multi-output geometry outside float32 support")
        return result


def _result(lower, upper, code, unchanged):
    method = "multi-squared-polynomial-interval-v1"
    if code not in (0, 2, 3) or unchanged not in (0, 1):
        raise RuntimeError("invalid device comparison summary")
    if code == 3:
        raise ValueError("multi-output geometry outside float32 support")
    if code == 2:
        return LossChange(None, None, method, "arithmetic_range")
    if unchanged:
        return LossChange(0, 0, method, "identical_stored_raw", unchanged=True)
    return LossChange(
        lower, upper, method, "contains_zero" if lower <= 0 <= upper else "bounded_sign"
    )


@_atomic
def compare(ops, problem, before, after):
    """Enclose the exact polynomial change; validate all channels before shortcuts."""
    target, offset = _arrays(ops, problem)
    shape = (problem.data.n_rows, problem.raw_width)
    old, new = ops._float(before, shape), ops._float(after, shape)
    ops._validate(old)
    ops._validate(new)
    with _workspace(ops):
        rows = ops.execution._empty((shape[0], 4), np.float64)
        output = ops.execution._empty((4,), np.float64)
        ops.execution._counts.setdefault("comparison_calls", 0)
        ops.execution._counts["comparison_calls"] += 1
        ops._launch(
            "multi_squared_compare_rows",
            shape[0],
            target,
            offset,
            old,
            new,
            ops.execution._array(rows),
        )
        ops._launch(
            "glm_compare_reduce",
            1,
            ops.execution._array(rows),
            ops.execution._array(problem.data.weight),
            ops.execution._array(output),
        )
        summary = ops.execution.export(output)
        ops.execution._counts.setdefault("comparison_export_bytes", 0)
        ops.execution._counts["comparison_export_bytes"] += output.nbytes
        return _result(*(float(v) for v in summary))
