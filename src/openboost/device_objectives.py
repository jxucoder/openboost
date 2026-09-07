"""Experimental resident scalar squared geometry; real-device acceptance pending."""

from dataclasses import dataclass

import numpy as np

from .device import DeviceData, _atomic, _workspace
from .objectives import Squared


@dataclass(frozen=True, eq=False)
class DeviceProblem:
    """Prepared scalar targets/offsets, privately owned by one operations instance."""

    data: DeviceData


@_atomic
def prepare(ops, data, problem):
    """Explicit target/offset upload; reuse the matching prepared data and weights."""
    Squared.validate(problem)
    ops._get(data, DeviceData)
    if data.problem_identity != problem.identity:
        raise ValueError("prepared problem identity differs")
    with np.errstate(over="raise", invalid="raise"):
        host = tuple(a.astype(np.float32) for a in (problem.target, problem.offset))
    if any(not np.isfinite(a).all() for a in host):
        raise ValueError("finite float32 targets and offsets required")
    handles = tuple(ops.execution.upload(a) for a in host)
    return ops._record(DeviceProblem(data), handles, handles)


def _arrays(ops, problem):
    ops._get(problem, DeviceProblem)
    return tuple(ops.execution._array(h) for h in ops._records[problem][0])


@_atomic
def base(ops, problem):
    """Weighted mean of target minus offset; return an owned float32 [1] buffer."""
    target, offset = _arrays(ops, problem)
    context = ops.execution
    output = context._empty((1,), np.float32)
    ops._launch(
        "squared_base",
        1,
        target,
        offset,
        context._array(problem.data.weight),
        context._array(output),
    )
    ops._validate(context._array(output).reshape(1, 1))
    return output


@_atomic
def broadcast(ops, value, n_rows):
    """Broadcast an owned float32 scalar into a new resident [N, 1] raw buffer."""
    if type(n_rows) is not int or not 0 < n_rows <= np.iinfo(np.int32).max:
        raise ValueError("positive int32 row count required")
    source = ops._float(value, (1,))
    ops._validate(source.reshape(1, 1))
    output = ops.execution._empty((n_rows, 1), np.float32)
    ops._launch("scalar_broadcast", n_rows, source, ops.execution._array(output))
    return output


@_atomic
def gradient(ops, problem, raw):
    """Unweighted resident gradient [N]; offset belongs to the objective only."""
    target, offset = _arrays(ops, problem)
    values = ops._float(raw, (problem.data.n_rows, 1))
    context = ops.execution
    output = context._empty((problem.data.n_rows,), np.float32)
    ops._launch(
        "squared_gradient", problem.data.n_rows, target, offset, values, context._array(output)
    )
    ops._validate(context._array(output).reshape(-1, 1))
    return output


@_atomic
def fields(ops, problem, raw):
    """Named scalar gradient/curvature with objective weight applied exactly once."""
    with _workspace(ops) as retained:
        target, offset = _arrays(ops, problem)
        values = ops._float(raw, (problem.data.n_rows, 1))
        context = ops.execution
        output = context._empty((problem.data.n_rows, 2), np.float32)
        ops._launch(
            "squared_fields", problem.data.n_rows, target, offset, values, context._array(output)
        )
        unweighted = ops.fields(
            problem.data,
            output,
            names=("gradient", "curvature"),
            roles=("unweighted", "unweighted"),
        )
        weighted = ops.apply_weight(unweighted)
        retained.add(weighted)
        return weighted


@_atomic
def loss(ops, problem, raw):
    """Ordered float64 weighted loss reduction; export only the final scalar."""
    target, offset = _arrays(ops, problem)
    values = ops._float(raw, (problem.data.n_rows, 1))
    ops._validate(values)
    context = ops.execution
    output = context._empty((1,), np.float64)
    ops._launch(
        "squared_loss",
        1,
        target,
        offset,
        context._array(problem.data.weight),
        values,
        context._array(output),
    )
    result = float(context.export(output)[0])
    context._counts.setdefault("metric_export_bytes", 0)
    context._counts["metric_export_bytes"] += output.nbytes
    context.release(output)
    if not np.isfinite(result):
        raise ValueError("finite scalar loss required")
    return result
