"""Experimental resident fixed-scale event/right-censored log-normal operations."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from . import device_objectives as operations
from .comparison import _aft_result
from .device import DeviceData, _atomic, _workspace
from .survival import AFTModel, LogNormalAFT, _scale


@dataclass(frozen=True, eq=False)
class AFTProblem(operations.DeviceProblem):
    """Owned lower times, offsets and censoring flags, bound to one fixed scale."""

    sigma: float = 1.0


def _configuration(sigma):
    if not isinstance(sigma, (int, float, np.integer, np.floating)) or isinstance(sigma, (bool, np.bool_)):
        raise ValueError("positive finite float64 AFT scale required")
    return _scale(sigma)


def objective(sigma=1.0):
    """Compose fixed-scale AFT geometry and explicit loss-change callbacks."""
    sigma = _configuration(sigma)
    return operations.ObjectiveOperations(
        LogNormalAFT.validate,
        partial(prepare, sigma=sigma),
        partial(base, sigma=sigma),
        partial(loss, sigma=sigma),
        partial(gradient, sigma=sigma),
        partial(fields, sigma=sigma),
        partial(compare, sigma=sigma),
    )


def _host_arrays(problem):
    LogNormalAFT.validate(problem)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            lower, offset = (a.astype(np.float32) for a in (problem.target[:, :1], problem.offset))
        except FloatingPointError as error:
            raise ValueError("finite float32 lower times and offsets required") from error
    if not np.isfinite(lower).all() or not np.isfinite(offset).all() or np.any(lower <= 0):
        raise ValueError("positive finite float32 lower times and finite offsets required")
    event = problem.target[:, 0] == problem.target[:, 1]
    return lower, offset, event


@_atomic
def prepare(ops, data, problem, *, sigma=1.0):
    """Explicit upload; encode original censoring before discarding upper infinity."""
    sigma, host = _configuration(sigma), _host_arrays(problem)
    ops._get(data, DeviceData)
    if data.problem_identity != problem.identity:
        raise ValueError("prepared problem identity differs")
    handles = tuple(ops.execution.upload(a) for a in host)
    return ops._record(AFTProblem(data, target_width=2, sigma=sigma), handles, handles)


def _arrays(ops, problem, sigma=None):
    ops._get(problem, AFTProblem)
    if sigma is not None and _configuration(sigma) != problem.sigma:
        raise ValueError("prepared AFT scale differs from objective")
    return tuple(ops.execution._array(h) for h in ops._records[problem][0])


@_atomic
def base(ops, problem, *, sigma=None):
    """Weighted log lower time minus offset, then validate every initialized row."""
    lower, offset, _ = _arrays(ops, problem, sigma)
    context = ops.execution
    with _workspace(ops) as retained:
        output = context._empty((1,), np.float32)
        ops._launch("aft_base", 1, lower, offset, context._array(problem.data.weight), context._array(output))
        ops._validate(context._array(output).reshape(1, 1))
        raw = operations.broadcast(ops, output, problem.data.n_rows)
        geometry(ops, problem, raw)
        retained.add(output)
        return output


@_atomic
def geometry(ops, problem, raw, *, sigma=None):
    """Owned unweighted float32 [N,2] gradients/curvatures at stored inputs."""
    arrays = _arrays(ops, problem, sigma)
    values = ops._float(raw, (problem.data.n_rows, 1))
    ops._validate(values)
    output = ops.execution._empty((problem.data.n_rows, 2), np.float32)
    ops._launch("aft_geometry", problem.data.n_rows, *arrays, problem.sigma, values, ops.execution._array(output))
    ops._validate(ops.execution._array(output), message="AFT geometry outside float32 support")
    return output


@_atomic
def gradient(ops, problem, raw, *, sigma=None):
    """Owned unweighted float32 [N] gradient for the shared scalar runtime."""
    with _workspace(ops) as retained:
        matrix = geometry(ops, problem, raw, sigma=sigma)
        output = ops.execution._empty((problem.data.n_rows,), np.float32)
        ops._launch("glm_gradient", problem.data.n_rows, ops.execution._array(matrix), ops.execution._array(output))
        retained.add(output)
        return output


@_atomic
def fields(ops, problem, raw, *, sigma=None):
    """Named gradient/curvature fields, training weights applied exactly once."""
    with _workspace(ops) as retained:
        matrix = geometry(ops, problem, raw, sigma=sigma)
        unweighted = ops.fields(problem.data, matrix, names=("gradient", "curvature"), roles=("unweighted", "unweighted"))
        weighted = ops.apply_weight(unweighted)
        retained.add(weighted)
        return weighted


@_atomic
def loss(ops, problem, raw, *, sigma=None):
    """Weighted mean likelihood; validate all rows and export one float64 metric."""
    arrays = _arrays(ops, problem, sigma)
    values = ops._float(raw, (problem.data.n_rows, 1))
    ops._validate(values)
    context = ops.execution
    with _workspace(ops):
        output = context._empty((1,), np.float64)
        ops._launch("aft_loss", 1, *arrays, problem.sigma, context._array(problem.data.weight), values, context._array(output))
        result = float(context.export(output)[0])
        context._counts.setdefault("metric_export_bytes", 0)
        context._counts["metric_export_bytes"] += output.nbytes
        if not np.isfinite(result):
            raise ValueError("AFT geometry outside float32 support")
        return result


def export(run, state, *, best=False):
    """Export CPU inference with the prepared fitting scale; no caller-supplied scale."""
    from .device_runtime import DeviceRun

    if not isinstance(run, DeviceRun):
        raise ValueError("AFT DeviceRun required")
    prepared = run.problem
    _arrays(run.ops, prepared)
    _arrays(run.ops, run.validation_problem, prepared.sigma)
    return AFTModel(run.export(state, best=best), prepared.sigma)


@_atomic
def compare(ops, problem, before, after, *, sigma=None):
    """Enclose the stored-input likelihood change; export a 32-byte summary only.

    Every row's full geometry domain is checked before identical/zero-weight
    shortcuts. Snapshots remain caller-owned and all comparison scratch is released.
    No reporting-loss subtraction, host geometry or empirical epsilon supplies sign.
    """
    arrays = _arrays(ops, problem, sigma)
    shape = (problem.data.n_rows, 1)
    old, new = ops._float(before, shape), ops._float(after, shape)
    ops._validate(old)
    ops._validate(new)
    context = ops.execution
    with _workspace(ops):
        rows = context._empty((shape[0], 4), np.float64)
        output = context._empty((4,), np.float64)
        context._counts.setdefault("comparison_calls", 0)
        context._counts["comparison_calls"] += 1
        ops._launch("aft_compare_rows", shape[0], *arrays, problem.sigma, old, new, context._array(rows))
        ops._launch("glm_compare_reduce", 1, context._array(rows),
                    context._array(problem.data.weight), context._array(output))
        summary = context.export(output)
        context._counts.setdefault("comparison_export_bytes", 0)
        context._counts["comparison_export_bytes"] += output.nbytes
        lo, hi, code, unchanged = (float(value) for value in summary)
        if code == 3:
            raise ValueError("AFT geometry outside float32 support")
        if code not in (0, 1, 2) or unchanged not in (0, 1):
            raise RuntimeError("invalid device comparison summary")
        return _aft_result(lo, hi, int(code), bool(unchanged))
