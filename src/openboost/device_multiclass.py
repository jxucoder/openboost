"""Experimental resident softmax components; real-CUDA validation is pending."""

from dataclasses import dataclass

import numpy as np

from . import device_objectives as operations
from .comparison import _multiclass_result
from .device import DeviceData, _atomic, _workspace
from .objectives import Multiclass


@dataclass(frozen=True, eq=False)
class MulticlassProblem(operations.DeviceProblem):
    """Family-bound, owned encoded targets and K-column offsets."""


def objective():
    """Softmax dependencies for mapped scalar trees and explicit loss comparisons.

    Use geometry and channel_fields to fit K trees from one snapshot. Comparison
    bounds have local mathematical evidence; real-device validation is pending.
    """
    return operations.ObjectiveOperations(Multiclass.validate, prepare, base, loss, compare=compare)


def _host_arrays(problem):
    Multiclass.validate(problem)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            host = tuple(a.astype(np.float32) for a in (problem.target, problem.offset))
        except FloatingPointError as error:
            raise ValueError("finite float32 multiclass inputs required") from error
    if any(not np.isfinite(a).all() for a in host):
        raise ValueError("finite float32 multiclass inputs required")
    if not np.array_equal(host[0].astype(np.float64), problem.target):
        raise ValueError("class codes must be exactly representable in float32")
    return host


@_atomic
def prepare(ops, data, problem):
    """Upload aligned targets/offsets once, preserving the declared raw width."""
    host = _host_arrays(problem)
    ops._get(data, DeviceData)
    if data.problem_identity != problem.identity:
        raise ValueError("prepared problem identity differs")
    handles = tuple(ops.execution.upload(a) for a in host)
    return ops._record(MulticlassProblem(data, 1, problem.raw_width), handles, handles)


def _arrays(ops, problem):
    ops._get(problem, operations.DeviceProblem)
    if type(problem) is not MulticlassProblem:
        raise ValueError("prepared objective family differs")
    if problem.target_width != 1 or problem.raw_width < 2:
        raise ValueError("multiclass target/raw widths required")
    return tuple(ops.execution._array(h) for h in ops._records[problem][0])


@_atomic
def base(ops, problem):
    """K zero logits; require every declared class and supported offset geometry."""
    target, _ = _arrays(ops, problem)
    with _workspace(ops) as retained:
        output = ops.execution._empty((problem.raw_width,), np.float32)
        ops._launch("multiclass_base", problem.raw_width, target, ops.execution._array(output))
        ops._validate(ops.execution._array(output).reshape(1, -1))
        raw = operations.broadcast(ops, output, problem.data.n_rows)
        geometry(ops, problem, raw)
        retained.add(output)
        return output


@_atomic
def geometry(ops, problem, raw):
    """Owned unweighted [N,K] gradient and 2p(1-p) diagonal upper bound.

    Float64 arithmetic at stored inputs preserves dominant-class complements;
    every bound must remain positive finite in float32, including weight-zero rows.
    """
    target, offset = _arrays(ops, problem)
    values = ops._float(raw, (problem.data.n_rows, problem.raw_width))
    ops._validate(values)
    g, h = (ops.execution._empty(values.shape, np.float32) for _ in range(2))
    ops._launch(
        "multiclass_geometry",
        problem.data.n_rows,
        target,
        offset,
        values,
        ops.execution._array(g),
        ops.execution._array(h),
    )
    for output in (g, h):
        ops._validate(
            ops.execution._array(output), message="multiclass geometry outside float32 support"
        )
    return g, h


def _channel(channel, width):
    if type(channel) is not int or not 0 <= channel < width:
        raise ValueError("valid integer class channel required")
    return channel


@_atomic
def channel_fields(ops, data, gradient, bound, *, channel):
    """Select one class from a shared geometry snapshot, applying weights once.

    Input matrices remain caller-owned. The returned named fields own their
    storage and use the existing scalar histogram/split/leaf operations.
    """
    ops._get(data, DeviceData)
    g = ops.execution._array(gradient)
    if g.ndim != 2 or g.shape[0] != data.n_rows or g.shape[1] < 2:
        raise ValueError("aligned multiclass geometry matrix required")
    channel = _channel(channel, g.shape[1])
    g, h = ops._float(gradient, g.shape), ops._float(bound, g.shape)
    ops._validate(g)
    ops._validate(h, nonnegative=True)
    with _workspace(ops) as retained:
        output = ops.execution._empty((data.n_rows, 2), np.float32)
        ops._launch("multiclass_fields", data.n_rows, g, h, channel, ops.execution._array(output))
        fields = ops.fields(
            data, output, names=("gradient", "curvature"), roles=("unweighted", "unweighted")
        )
        weighted = ops.apply_weight(fields)
        retained.add(weighted)
        return weighted


@_atomic
def loss(ops, problem, raw):
    """Weighted mean softmax NLL; export only one float64 reporting metric."""
    target, offset = _arrays(ops, problem)
    values = ops._float(raw, (problem.data.n_rows, problem.raw_width))
    ops._validate(values)
    with _workspace(ops):
        output = ops.execution._empty((1,), np.float64)
        ops._launch(
            "multiclass_loss",
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
            raise ValueError("multiclass geometry outside float32 support")
        return result


@_atomic
def compare(ops, problem, before, after):
    """Bound mean NLL change on CUDA; only a 32-byte summary leaves the device.

    Every class of both snapshots is validated before identity/weight shortcuts.
    Caller snapshots remain independent and owned. No reported-loss subtraction,
    host objective calculation or training-array export supplies the decision.
    """
    target, offset = _arrays(ops, problem)
    shape = (problem.data.n_rows, problem.raw_width)
    old, new = ops._float(before, shape), ops._float(after, shape)
    ops._validate(old)
    ops._validate(new)
    context = ops.execution
    with _workspace(ops):
        rows = context._empty((shape[0], 4), np.float64)
        output = context._empty((4,), np.float64)
        context._counts.setdefault("comparison_calls", 0)
        context._counts["comparison_calls"] += 1
        ops._launch(
            "multiclass_compare_rows", shape[0], target, offset, old, new, context._array(rows)
        )
        ops._launch(
            "glm_compare_reduce",
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
            raise ValueError("multiclass geometry outside float32 support")
        if code not in (0, 1, 2) or unchanged not in (0, 1):
            raise RuntimeError("invalid device comparison summary")
        return _multiclass_result(lower, upper, int(code), bool(unchanged))
