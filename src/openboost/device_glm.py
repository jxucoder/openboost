"""Experimental resident binary and Poisson components; CUDA validation pending."""

from dataclasses import dataclass
from functools import partial

import numpy as np

from . import device_objectives as operations
from .comparison import _glm_result
from .device import DeviceData, _atomic, _parameter, _workspace
from .objectives import Binary, Poisson


@dataclass(frozen=True, eq=False)
class GLMProblem(operations.DeviceProblem):
    """Owned scalar targets/offsets and, for Poisson, separate exposure storage."""

    family: str = "binary"


def _configuration(family, value):
    value = _parameter(value)
    if family == "binary":
        if not 0 < value < 0.5 or 1 - float(value) == 1:
            raise ValueError("representable probability clip in (0, 0.5) required")
    elif family == "poisson":
        if value <= 0:
            raise ValueError("positive finite float32 minimum rate required")
    else:
        raise ValueError("binary or poisson family required")
    return value


def binary(*, clip=1e-6):
    """Binary callbacks with an offset-centred prior and stable logistic tails.

    The convex loss-change callback has local arithmetic checks; real-device
    comparison and recipe validation remain pending.
    """
    return _objective("binary", clip)


def poisson(*, minimum_rate=1e-6):
    """Exposure-aware count callbacks; minimum rate applies to zero-count base only."""
    return _objective("poisson", minimum_rate)


def _objective(family, parameter):
    parameter = _configuration(family, parameter)
    return operations.ObjectiveOperations(
        Binary.validate if family == "binary" else Poisson.validate,
        partial(prepare, family=family),
        partial(base, family=family, parameter=parameter),
        partial(loss, family=family),
        partial(gradient, family=family),
        partial(fields, family=family),
        partial(compare, family=family),
    )


def _host_arrays(problem, family):
    """Validate the explicit upload boundary; never compute training geometry here."""
    if family not in ("binary", "poisson"):
        raise ValueError("binary or poisson family required")
    (Binary.validate if family == "binary" else Poisson.validate)(problem)
    source = (problem.target, problem.offset)
    if family == "poisson":
        source += (problem.structure["exposure"],)
    with np.errstate(over="raise", invalid="raise", under="ignore"):
        try:
            host = tuple(a.astype(np.float32) for a in source)
        except FloatingPointError as error:
            raise ValueError("finite float32 objective inputs required") from error
    if any(not np.isfinite(a).all() for a in host):
        raise ValueError("finite float32 objective inputs required")
    if family == "poisson":
        if not np.array_equal(host[0].astype(np.float64), problem.target):
            raise ValueError("Poisson counts must be exactly representable in float32")
        if np.any(host[2] <= 0):
            raise ValueError("positive float32 exposure required")
    return host


@_atomic
def prepare(ops, data, problem, *, family):
    """Upload aligned targets/offsets/exposure once and bind their objective family."""
    host = _host_arrays(problem, family)
    ops._get(data, DeviceData)
    if data.problem_identity != problem.identity:
        raise ValueError("prepared problem identity differs")
    handles = tuple(ops.execution.upload(a) for a in host)
    return ops._record(GLMProblem(data, family=family), handles, handles)


def _arrays(ops, problem, family):
    ops._get(problem, GLMProblem)
    if family not in ("binary", "poisson") or problem.family != family:
        raise ValueError("prepared objective family differs")
    return tuple(ops.execution._array(h) for h in ops._records[problem][0])


@_atomic
def base(ops, problem, *, family, parameter=1e-6):
    """Resident float64 reduction, stored as float32; validate every initialized row."""
    parameter = _configuration(family, parameter)
    arrays = _arrays(ops, problem, family)
    with _workspace(ops) as retained:
        output = ops.execution._empty((1,), np.float32)
        ops._launch(
            family + "_base",
            1,
            *arrays,
            ops.execution._array(problem.data.weight),
            parameter,
            ops.execution._array(output),
        )
        ops._validate(ops.execution._array(output).reshape(1, 1))
        raw = operations.broadcast(ops, output, problem.data.n_rows)
        geometry(ops, problem, raw, family=family)
        retained.add(output)
        return output


@_atomic
def geometry(ops, problem, raw, *, family):
    """Owned unweighted float32 [N,2] gradient/curvature with positive curvature."""
    arrays = _arrays(ops, problem, family)
    values = ops._float(raw, (problem.data.n_rows, 1))
    ops._validate(values)
    output = ops.execution._empty((problem.data.n_rows, 2), np.float32)
    ops._launch(
        family + "_geometry",
        problem.data.n_rows,
        *arrays,
        values,
        ops.execution._array(output),
    )
    ops._validate(ops.execution._array(output), message="GLM geometry outside float32 support")
    return output


@_atomic
def gradient(ops, problem, raw, *, family):
    """Owned unweighted float32 [N] gradient for the shared scalar runtime."""
    with _workspace(ops) as retained:
        matrix = geometry(ops, problem, raw, family=family)
        output = ops.execution._empty((problem.data.n_rows,), np.float32)
        ops._launch(
            "glm_gradient",
            problem.data.n_rows,
            ops.execution._array(matrix),
            ops.execution._array(output),
        )
        retained.add(output)
        return output


@_atomic
def fields(ops, problem, raw, *, family):
    """Named gradient/curvature with training weights applied exactly once."""
    with _workspace(ops) as retained:
        matrix = geometry(ops, problem, raw, family=family)
        unweighted = ops.fields(
            problem.data,
            matrix,
            names=("gradient", "curvature"),
            roles=("unweighted", "unweighted"),
        )
        weighted = ops.apply_weight(unweighted)
        retained.add(weighted)
        return weighted


@_atomic
def loss(ops, problem, raw, *, family):
    """Weighted mean likelihood at stored inputs; export only one float64 metric."""
    arrays = _arrays(ops, problem, family)
    values = ops._float(raw, (problem.data.n_rows, 1))
    ops._validate(values)
    with _workspace(ops):
        output = ops.execution._empty((1,), np.float64)
        ops._launch(
            family + "_loss",
            1,
            *arrays,
            ops.execution._array(problem.data.weight),
            values,
            ops.execution._array(output),
        )
        result = float(ops.execution.export(output)[0])
        ops.execution._counts.setdefault("metric_export_bytes", 0)
        ops.execution._counts["metric_export_bytes"] += output.nbytes
        if not np.isfinite(result):
            raise ValueError("GLM geometry outside float32 support")
        return result


@_atomic
def compare(ops, problem, before, after, *, family):
    """Resident convex loss-change bounds, with independent caller-owned snapshots.

    Validate every row before unchanged/zero-weight shortcuts. No reporting-loss
    subtraction or CPU computation occurs; only a 32-byte summary and validation
    flags leave CUDA. Finite but unresolvable changes retain explicit reasons.
    """
    arrays = _arrays(ops, problem, family)
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
        ops._launch(family + "_compare_rows", shape[0], *arrays, old, new, context._array(rows))
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
            raise ValueError("GLM geometry outside float32 support")
        if code not in (0, 1, 2) or unchanged not in (0, 1):
            raise RuntimeError("invalid device comparison summary")
        return _glm_result(family, lower, upper, int(code), bool(unchanged))
