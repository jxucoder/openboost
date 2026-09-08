"""Resident objective operations; new K-column composition awaits device validation."""

from dataclasses import dataclass

import numpy as np

from .comparison import LossChange
from .device import DeviceData, _atomic, _parameter, _workspace
from .objectives import Squared


@dataclass(frozen=True, eq=False)
class DeviceProblem:
    """Prepared targets/offsets, privately owned by one operations instance."""

    data: DeviceData
    target_width: int = 1
    raw_width: int = 1


@_atomic
def prepare(ops, data, problem, *, validate=Squared.validate):
    """Explicit target/offset upload; reuse the matching prepared data and weights."""
    validate(problem)
    ops._get(data, DeviceData)
    if data.problem_identity != problem.identity:
        raise ValueError("prepared problem identity differs")
    with np.errstate(over="raise", invalid="raise"):
        host = tuple(a.astype(np.float32) for a in (problem.target, problem.offset))
    if any(not np.isfinite(a).all() for a in host):
        raise ValueError("finite float32 targets and offsets required")
    handles = tuple(ops.execution.upload(a) for a in host)
    return ops._record(
        DeviceProblem(data, problem.target.shape[1], problem.raw_width), handles, handles
    )


def _arrays(ops, problem, *, widths=(1, 1)):
    ops._get(problem, DeviceProblem)
    if (problem.target_width, problem.raw_width) != widths:
        raise ValueError("prepared target/raw widths differ from objective")
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
    """Broadcast a float32 [K] base into an owned resident [N,K] raw buffer."""
    if type(n_rows) is not int or not 0 < n_rows <= np.iinfo(np.int32).max:
        raise ValueError("positive int32 row count required")
    source = ops.execution._array(value)
    if source.ndim != 1 or not 0 < source.size <= np.iinfo(np.int32).max:
        raise ValueError("nonempty float32 base vector required")
    source = ops._float(value, source.shape)
    ops._validate(source.reshape(1, -1))
    output = ops.execution._empty((n_rows, source.size), np.float32)
    ops._launch("raw_broadcast", n_rows, source, ops.execution._array(output))
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


@dataclass(frozen=True)
class ObjectiveOperations:
    """Explicit objective dependencies; algorithm geometry stays outside runtime."""

    validate: object
    prepare: object
    base: object
    loss: object
    gradient: object = None
    fields: object = None
    compare: object = None

    def __post_init__(self):
        if any(not callable(f) for f in (self.validate, self.prepare, self.base, self.loss)):
            raise ValueError("callable validation/preparation/base/loss operations required")
        if any(
            f is not None and not callable(f) for f in (self.gradient, self.fields, self.compare)
        ):
            raise ValueError("optional gradient/fields/comparison operations must be callable")

    def loss_change(self, ops, problem, before, after):
        """Request the objective's explicit comparison; never subtract reported losses."""
        if self.compare is None:
            raise NotImplementedError("objective does not supply a loss-change operation")
        result = self.compare(ops, problem, before, after)
        if not isinstance(result, LossChange):
            raise TypeError("objective comparison must return LossChange")
        return result


SQUARED = ObjectiveOperations(Squared.validate, prepare, base, loss, gradient, fields)


def direction_configuration(mode, damping):
    damping = _parameter(damping)
    if mode not in ("ordinary", "natural") or (mode == "ordinary" and damping != 0):
        raise ValueError("ordinary (zero damping) or natural direction required")
    return mode, damping


@_atomic
def diagonal_direction(ops, gradient, metric, *, mode="natural", damping=0.0):
    """Unweighted [N,K] diagonal solve; no objective-specific branches or weights."""
    mode, damping = direction_configuration(mode, damping)
    g = ops.execution._array(gradient)
    if g.ndim != 2 or not all(g.shape):
        raise ValueError("nonempty [N,K] gradient required")
    g = ops._float(gradient, g.shape)
    h = ops._float(metric, g.shape)
    ops._validate(g)
    ops._validate(h, nonnegative=True)
    output = ops.execution._empty(g.shape, np.float32)
    ops._launch(
        "diagonal_direction",
        g.shape[0],
        g,
        h,
        mode == "natural",
        damping,
        ops.execution._array(output),
    )
    ops._validate(ops.execution._array(output))
    return output


@_atomic
def least_squares(ops, data, direction, channel):
    """Fit one direction column with G=-w*z and H=w, independent of Fisher."""
    ops._get(data, DeviceData)
    values = ops.execution._array(direction)
    if (
        values.ndim != 2
        or values.shape[0] != data.n_rows
        or type(channel) is not int
        or not 0 <= channel < values.shape[1]
    ):
        raise ValueError("aligned direction matrix and valid integer channel required")
    values = ops._float(direction, values.shape)
    ops._validate(values)
    with _workspace(ops) as retained:
        output = ops.execution._empty((data.n_rows, 2), np.float32)
        ops._launch("direction_fields", data.n_rows, values, channel, ops.execution._array(output))
        unweighted = ops.fields(
            data, output, names=("gradient", "curvature"), roles=("unweighted", "unweighted")
        )
        weighted = ops.apply_weight(unweighted)
        retained.add(weighted)
        return weighted
