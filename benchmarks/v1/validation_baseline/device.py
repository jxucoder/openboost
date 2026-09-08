"""Experimental public CUDA fields, histograms and split operations."""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from time import perf_counter

import numpy as np

from .binning import BinnedData
from .data import Problem
from .execution import DeviceBuffer, ExecutionContext


@dataclass(frozen=True, eq=False)
class DeviceData:
    data_identity: str
    prepared_identity: str
    problem_identity: str
    binning_identity: str
    n_rows: int
    feature_names: tuple[str, ...]
    bin_counts: tuple[int, ...]
    codes: DeviceBuffer
    missing: DeviceBuffer
    weight: DeviceBuffer


@dataclass(frozen=True, eq=False)
class DeviceFields:
    data: DeviceData
    names: tuple[str, ...]
    roles: tuple[str, ...]
    values: DeviceBuffer


@dataclass(frozen=True, eq=False)
class DeviceRows:
    data: DeviceData
    positions: DeviceBuffer


@dataclass(frozen=True, eq=False)
class DeviceHistogram:
    data: DeviceData
    fields: DeviceFields
    rows: DeviceRows
    sums: DeviceBuffer
    counts: DeviceBuffer
    total: DeviceBuffer


@dataclass(frozen=True, eq=False)
class DeviceCandidates:
    histogram: DeviceHistogram
    values: DeviceBuffer
    counts: DeviceBuffer
    active: DeviceBuffer

    @property
    def data(self):
        return self.histogram.data

    @property
    def size(self):
        return self.active.shape[0]

    def key(self, index):
        """Metadata for a padded slot; consult active before treating it as a candidate."""
        if (
            not isinstance(index, (int, np.integer))
            or isinstance(index, bool)
            or not 0 <= index < self.size
        ):
            raise ValueError("in-range integer candidate index required")
        width = max(self.data.bin_counts)
        return int(index // (2 * width)), int((index // 2) % width), bool(index % 2)


@dataclass(frozen=True, eq=False)
class DeviceScores:
    candidates: DeviceCandidates
    values: DeviceBuffer

    @property
    def data(self):
        return self.candidates.data


@dataclass(frozen=True, eq=False)
class DeviceMask:
    candidates: DeviceCandidates
    values: DeviceBuffer

    @property
    def data(self):
        return self.candidates.data


@dataclass(frozen=True, eq=False)
class DeviceSplit:
    candidates: DeviceCandidates
    index: int

    @property
    def data(self):
        return self.candidates.data

    @property
    def key(self):
        return self.candidates.key(self.index)


def _atomic(method):
    """Discard this operation's new buffers/records after allocation/validation failure."""

    @wraps(method)
    def call(self, *args, **kwargs):
        context = self.execution
        context._check()
        before, records = set(context._buffers), set(self._records)
        try:
            with context._scope():
                return method(self, *args, **kwargs)
        except BaseException:
            try:
                context.synchronize()
            finally:
                for handle in set(context._buffers) - before:
                    del context._buffers[handle]
                    context._counts["live_bytes"] -= handle.nbytes
                for record in set(self._records) - records:
                    del self._records[record]
            raise

    return call


@contextmanager
def _workspace(ops):
    """Release new scratch, retaining only explicitly returned records/buffers.

    Nested composition can allocate freely on the owned stream. Callers must not
    retain newly created callback scratch outside this scope. Existing inputs are
    untouched. On failure even designated outputs are discarded.
    """
    context = ops.execution
    context._check()
    before, records = set(context._buffers), set(ops._records)
    retained = set()
    try:
        yield retained
    except BaseException:
        retained.clear()
        raise
    finally:
        handles = {r for r in retained if isinstance(r, DeviceBuffer)}
        for record in retained - handles:
            handles.update(ops._records[record][0])
        context.synchronize()
        for handle in set(context._buffers) - before - handles:
            del context._buffers[handle]
            context._counts["live_bytes"] -= handle.nbytes
        for record in set(ops._records) - records - retained:
            del ops._records[record]


def _schema(names, roles, width):
    names, roles = tuple(names), tuple(roles)
    if (
        not names
        or len(names) != width
        or any(not isinstance(n, str) or not n for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("unique nonempty names must match fields")
    if len(roles) != width or any(
        r not in ("unweighted", "training", "independent") for r in roles
    ):
        raise ValueError("explicit field weight roles required")
    return names, roles


def _parameter(value):
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or value < 0
    ):
        raise ValueError("finite nonnegative float32 parameter required")
    with np.errstate(over="raise", invalid="raise"):
        try:
            result = np.float32(value)
        except (ValueError, TypeError, OverflowError, FloatingPointError) as error:
            raise ValueError("finite nonnegative float32 parameter required") from error
    if not np.isfinite(result):
        raise ValueError("finite nonnegative float32 parameter required")
    return result


class DeviceOperations:
    """Public composition of context-owned fields and actual routed reductions.

    Prepare uploads CPU binned inputs/weights once. Fields/rows may borrow opaque
    buffers from the same context; operations return independent output allocations.
    Releasing a borrowed buffer invalidates its record. No mutable array is exposed.
    Records are bound to this operations instance and cannot be forged with replace.
    """

    def __init__(self, execution):
        if not isinstance(execution, ExecutionContext):
            raise ValueError("ExecutionContext required")
        execution._check()
        self.execution = execution
        self._records = {}
        for key in (
            "kernel_launches",
            "kernel_dispatch_seconds",
            "validation_export_bytes",
            "decision_export_bytes",
        ):
            execution._counts.setdefault(key, 0)

    def _record(self, record, handles, owned=()):
        self._records[record] = (tuple(handles), tuple(owned))
        return record

    def _get(self, record, kind):
        self.execution._check()
        if not isinstance(record, kind) or record not in self._records:
            raise ValueError("foreign, forged or released device record")
        data = getattr(record, "data", None)
        if data is not None:
            self._get(data, DeviceData)
        for handle in self._records[record][0]:
            self.execution._array(handle)
        return record

    def _float(self, handle, shape):
        array = self.execution._array(handle)
        if array.dtype != np.dtype("float32") or tuple(array.shape) != shape:
            raise ValueError(f"aligned float32 buffer with shape {shape} required")
        return array

    def _launch(self, name, size, *args):
        from numba import cuda

        from . import _device_kernels

        context = self.execution
        start = perf_counter()
        # Every array is owned by this context and all work uses its one stream.
        # Avoid CAI's implicit per-argument synchronization; never import external arrays.
        with context._scope():
            stream = cuda.external_stream(context._stream.ptr)
            arrays = [
                cuda.as_cuda_array(a, sync=False) if isinstance(a, context._cp.ndarray) else a
                for a in args
            ]
            context._counts["kernel_launches"] += 1
            try:
                getattr(_device_kernels, name)[max(1, (size + 127) // 128), 128, stream](*arrays)
            finally:
                context._counts["kernel_dispatch_seconds"] += perf_counter() - start

    def _flags(self, handle, message):
        context = self.execution
        flags = context.export(handle)
        context._counts["validation_export_bytes"] += flags.nbytes
        context.release(handle)
        if flags.any():
            raise ValueError(message)

    def _validate(self, array, *, nonnegative=False, message=None):
        context = self.execution
        flags = context._empty((array.shape[1],), np.int32)
        self._launch("validate_fields", array.shape[1], array, nonnegative, context._array(flags))
        self._flags(
            flags,
            message
            or "finite fields required; declared nonnegative information cannot be negative",
        )

    @_atomic
    def prepare(self, binned, problem):
        """Explicit host preparation boundary, numeric features and float32 weights."""
        if (
            not isinstance(binned, BinnedData)
            or not isinstance(problem, Problem)
            or binned.data.identity != problem.data.identity
        ):
            raise ValueError("binned data and problem must share row identity")
        if any(k != "numeric" for k in binned.binning.feature_kinds):
            raise ValueError("device aggregation currently supports numeric features only")
        if len(problem.weight) > np.iinfo(np.int32).max:
            raise ValueError("int32 row capacity exceeded")
        with np.errstate(over="raise", invalid="raise"):
            weight = problem.weight.astype(np.float32)
        if not np.isfinite(weight).all() or not np.any(weight > 0):
            raise ValueError("weights must retain finite positive mass in float32")
        context = self.execution
        codes, missing, weights = (
            context.upload(v) for v in (binned.codes, binned.missing, weight)
        )
        record = DeviceData(
            problem.data.identity,
            binned.identity,
            problem.identity,
            binned.binning.identity,
            len(weight),
            binned.data.feature_names,
            binned.binning.bin_counts,
            codes,
            missing,
            weights,
        )
        return self._record(record, (codes, missing, weights), (codes, missing, weights))

    @_atomic
    def fields(self, data, values, *, names, roles):
        """Bind finite resident values to a prepared problem with explicit weight roles."""
        self._get(data, DeviceData)
        array = self.execution._array(values)
        if array.ndim != 2:
            raise ValueError("aligned float32 matrix required")
        names, roles = _schema(names, roles, array.shape[1])
        array = self._float(values, (data.n_rows, len(names)))
        self._validate(array)
        return self._record(DeviceFields(data, names, roles, values), (values,))

    @_atomic
    def apply_weight(self, fields):
        """Weight only unweighted columns once; independent columns remain unchanged."""
        self._get(fields, DeviceFields)
        if "training" in fields.roles or "unweighted" not in fields.roles:
            raise ValueError("training weight already applied or no unweighted fields")
        context = self.execution
        output = context._empty(fields.values.shape, np.float32)
        array = context._array(output)
        self._launch(
            "weight_fields",
            array.size,
            context._array(fields.values),
            context._array(fields.data.weight),
            tuple(r == "unweighted" for r in fields.roles),
            array,
        )
        self._validate(array)
        roles = tuple("training" if r == "unweighted" else r for r in fields.roles)
        return self._record(
            DeviceFields(fields.data, fields.names, roles, output), (output,), (output,)
        )

    @_atomic
    def add_independent(self, fields, name, values, *, nonnegative=False):
        """Append an independent resident column; D2 explicitly requests nonnegative."""
        self._get(fields, DeviceFields)
        names, roles = _schema(
            (*fields.names, name), (*fields.roles, "independent"), len(fields.names) + 1
        )
        if type(nonnegative) is not bool:
            raise ValueError("nonnegative must be boolean")
        column = self._float(values, (fields.data.n_rows,))
        self._validate(column.reshape(-1, 1), nonnegative=nonnegative)
        context = self.execution
        output = context._empty((fields.data.n_rows, len(names)), np.float32)
        self._launch(
            "append_field",
            fields.data.n_rows * len(names),
            context._array(fields.values),
            column,
            context._array(output),
        )
        return self._record(DeviceFields(fields.data, names, roles, output), (output,), (output,))

    @_atomic
    def rows(self, data, positions=None):
        """Original row positions, supplied explicitly on host or already resident."""
        self._get(data, DeviceData)
        context = self.execution
        owned = ()
        if positions is None:
            positions = context._empty((data.n_rows,), np.int32)
            self._launch("row_positions", data.n_rows, context._array(positions))
            return self._record(DeviceRows(data, positions), (positions,), (positions,))
        if isinstance(positions, DeviceBuffer):
            array = context._array(positions)
            if array.ndim != 1 or array.dtype != np.dtype("int32"):
                raise ValueError("one-dimensional int32 row positions required")
            if array.size > data.n_rows:
                raise ValueError("unique in-range row positions required")
            seen = context._empty((data.n_rows,), np.uint8)
            flags = context._empty((1,), np.int32)
            self._launch("validate_rows", 1, array, context._array(seen), context._array(flags))
            self._flags(flags, "unique in-range row positions required")
            context.release(seen)
        else:
            if hasattr(positions, "__cuda_array_interface__"):
                raise ValueError("context-owned row buffer required")
            selected = np.asarray(positions)
            if (
                selected.ndim != 1
                or (selected.size and selected.dtype.kind not in "iu")
                or np.any(selected < 0)
                or np.any(selected >= data.n_rows)
                or len(np.unique(selected)) != len(selected)
            ):
                raise ValueError("unique in-range row positions required")
            positions = (
                context.upload(selected.astype(np.int32))
                if selected.size
                else context._empty((0,), np.int32)
            )
            owned = (positions,)
        return self._record(DeviceRows(data, positions), (positions,), owned)

    @_atomic
    def histogram(self, data, fields, rows):
        """Padded [feature, bin, field] sums, integer counts and original-row total.

        Each feature's missing bin is at its own bin_count; higher padding is zero.
        No weight is applied here. Only compact finite-value flags return to host.
        """
        self._get(data, DeviceData)
        self._get(fields, DeviceFields)
        self._get(rows, DeviceRows)
        if fields.data is not data or rows.data is not data:
            raise ValueError("fields and rows must share the prepared problem identity")
        if "unweighted" in fields.roles:
            raise ValueError("apply objective training weights before aggregation")
        context = self.execution
        shape = (len(data.bin_counts), max(data.bin_counts) + 1, len(fields.names))
        sums = context._empty(shape, np.float32)
        counts = context._empty(shape[:2], np.int64)
        total = context._empty((shape[2],), np.float32)
        self._launch(
            "histogram",
            int(np.prod(shape)),
            context._array(data.codes),
            context._array(data.missing),
            data.bin_counts,
            context._array(fields.values),
            context._array(rows.positions),
            context._array(sums),
            context._array(counts),
        )
        self._launch(
            "row_total",
            shape[2],
            context._array(fields.values),
            context._array(rows.positions),
            context._array(total),
        )
        self._validate(context._array(sums).reshape(-1, shape[2]))
        self._validate(context._array(total).reshape(1, -1))
        record = DeviceHistogram(data, fields, rows, sums, counts, total)
        return self._record(record, (sums, counts, total), (sums, counts, total))

    def _batch(self, batch):
        self._get(batch, DeviceCandidates)
        self._get(batch.histogram, DeviceHistogram)
        self._get(batch.histogram.fields, DeviceFields)
        self._get(batch.histogram.rows, DeviceRows)
        return batch

    def _newton_columns(self, histogram):
        fields = self._get(histogram.fields, DeviceFields)
        if any(name not in fields.names for name in ("gradient", "curvature")):
            raise ValueError("named scalar gradient and curvature required")
        g, h = fields.names.index("gradient"), fields.names.index("curvature")
        if fields.roles[g] != "training" or fields.roles[h] != "training":
            raise ValueError("once-weighted training gradient and curvature required")
        self._validate(self.execution._array(fields.values)[:, h : h + 1], nonnegative=True)
        return g, h

    def _compact(self, buffer):
        result = self.execution.export(buffer)
        self.execution._counts["decision_export_bytes"] += result.nbytes
        self.execution.release(buffer)
        return result

    @_atomic
    def candidates(self, histogram):
        """Resident padded candidates; active bins match the full prepared CPU universe."""
        hist = self._get(histogram, DeviceHistogram)
        self._get(hist.fields, DeviceFields)
        self._get(hist.rows, DeviceRows)
        context = self.execution
        size = len(hist.data.bin_counts) * max(hist.data.bin_counts) * 2
        if size > np.iinfo(np.int32).max:
            raise ValueError("int32 candidate capacity exceeded")
        width = len(hist.fields.names)
        values = context._empty((size, 2, width), np.float32)
        counts = context._empty((size, 2), np.int64)
        active = context._empty((size,), bool)
        self._launch(
            "candidate_sums",
            size * width,
            context._array(hist.data.codes),
            context._array(hist.data.missing),
            hist.data.bin_counts,
            context._array(hist.sums),
            context._array(hist.counts),
            context._array(values),
            context._array(counts),
            context._array(active),
        )
        self._validate(context._array(values).reshape(-1, width))
        return self._record(
            DeviceCandidates(hist, values, counts, active),
            (values, counts, active),
            (values, counts, active),
        )

    @_atomic
    def scores(self, candidates, values):
        """Bind an explicitly supplied finite resident score vector to one batch."""
        batch = self._batch(candidates)
        array = self._float(values, (batch.size,))
        self._validate(array.reshape(-1, 1))
        return self._record(DeviceScores(batch, values), (values,))

    @_atomic
    def mask(self, candidates, values):
        """Bind an explicitly supplied resident bool vector to one batch."""
        batch = self._batch(candidates)
        array = self.execution._array(values)
        if array.dtype != np.dtype(bool) or array.shape != (batch.size,):
            raise ValueError("candidate-aligned bool mask required")
        return self._record(DeviceMask(batch, values), (values,))

    @_atomic
    def newton_scores(self, candidates, *, reg_lambda=1.0, split_penalty=0.0):
        batch = self._batch(candidates)
        regularization, penalty = _parameter(reg_lambda), _parameter(split_penalty)
        g, h = self._newton_columns(batch.histogram)
        context = self.execution
        output = context._empty((batch.size,), np.float32)
        self._launch(
            "scalar_scores",
            batch.size,
            context._array(batch.values),
            context._array(batch.counts),
            context._array(batch.active),
            context._array(batch.histogram.total),
            g,
            h,
            regularization,
            penalty,
            context._array(output),
        )
        self._validate(
            context._array(output).reshape(-1, 1),
            message="finite gains and positive finite Newton denominators required",
        )
        return self._record(DeviceScores(batch, output), (output,), (output,))

    @_atomic
    def feasible(self, candidates, *, min_child_h=0.0):
        batch = self._batch(candidates)
        minimum = _parameter(min_child_h)
        _, h = self._newton_columns(batch.histogram)
        context = self.execution
        output = context._empty((batch.size,), bool)
        self._launch(
            "scalar_feasible",
            batch.size,
            context._array(batch.values),
            context._array(batch.counts),
            context._array(batch.active),
            h,
            minimum,
            context._array(output),
        )
        return self._record(DeviceMask(batch, output), (output,), (output,))

    @_atomic
    def nonempty(self, candidates):
        """Structural two-child mask, independent of an objective's curvature rules."""
        batch = self._batch(candidates)
        context = self.execution
        output = context._empty((batch.size,), bool)
        self._launch(
            "candidate_nonempty",
            batch.size,
            context._array(batch.counts),
            context._array(batch.active),
            context._array(output),
        )
        return self._record(DeviceMask(batch, output), (output,), (output,))

    @_atomic
    def child_minimum(self, candidates, name, minimum):
        """Both children meet one named independent-information minimum."""
        batch = self._batch(candidates)
        minimum = _parameter(minimum)
        fields = batch.histogram.fields
        if name not in fields.names or fields.roles[fields.names.index(name)] != "independent":
            raise ValueError("named independent information field required")
        q = fields.names.index(name)
        context = self.execution
        self._validate(context._array(fields.values)[:, q : q + 1], nonnegative=True)
        output = context._empty((batch.size,), bool)
        self._launch(
            "information_minimum",
            batch.size,
            context._array(batch.values),
            context._array(batch.active),
            q,
            minimum,
            context._array(output),
        )
        return self._record(DeviceMask(batch, output), (output,), (output,))

    @_atomic
    def mask_and(self, left, right):
        self._get(left, DeviceMask)
        self._get(right, DeviceMask)
        batch = self._batch(left.candidates)
        if right.candidates is not batch:
            raise ValueError("masks must belong to the same candidate batch")
        context = self.execution
        output = context._empty((batch.size,), bool)
        self._launch(
            "combine_masks",
            batch.size,
            context._array(left.values),
            context._array(right.values),
            context._array(output),
        )
        return self._record(DeviceMask(batch, output), (output,), (output,))

    @_atomic
    def choose(self, candidates, scores, mask):
        """Best strictly positive masked gain; exact ties keep the first active key."""
        batch = self._batch(candidates)
        self._get(scores, DeviceScores)
        self._get(mask, DeviceMask)
        if scores.candidates is not batch or mask.candidates is not batch:
            raise ValueError("scores and mask must belong to the same candidate batch")
        context = self.execution
        output = context._empty((1,), np.int32)
        self._launch(
            "choose_candidate",
            1,
            context._array(scores.values),
            context._array(mask.values),
            context._array(batch.active),
            context._array(output),
        )
        index = int(self._compact(output)[0])
        if index == -1:
            return None
        if not 0 <= index < batch.size:
            raise RuntimeError("invalid device winner index")
        return self._record(DeviceSplit(batch, index), ())

    @_atomic
    def partition(self, rows, split):
        """Stable original-row child views; only child sizes return to the host."""
        self._get(rows, DeviceRows)
        self._get(split, DeviceSplit)
        batch = self._batch(split.candidates)
        if rows is not batch.histogram.rows:
            raise ValueError("split belongs to different routed rows")
        context = self.execution
        route = context._empty(rows.positions.shape, bool)
        sizes = context._empty((2,), np.int32)
        f, threshold, missing_left = split.key
        self._launch(
            "split_routes",
            1,
            context._array(rows.data.codes),
            context._array(rows.data.missing),
            context._array(rows.positions),
            f,
            threshold,
            missing_left,
            context._array(route),
            context._array(sizes),
        )
        left_size, right_size = (int(n) for n in self._compact(sizes))
        if min(left_size, right_size) < 0 or left_size + right_size != rows.positions.shape[0]:
            raise RuntimeError("invalid device child sizes")
        left = context._empty((left_size,), np.int32)
        right = context._empty((right_size,), np.int32)
        self._launch(
            "split_rows",
            1,
            context._array(rows.positions),
            context._array(route),
            context._array(left),
            context._array(right),
        )
        context.release(route)
        return tuple(self._record(DeviceRows(rows.data, b), (b,), (b,)) for b in (left, right))

    @_atomic
    def leaf(self, histogram, *, reg_lambda=1.0):
        """Owned scalar Newton value for exactly the histogram's routed rows."""
        hist = self._get(histogram, DeviceHistogram)
        self._get(hist.rows, DeviceRows)
        regularization = _parameter(reg_lambda)
        g, h = self._newton_columns(hist)
        context = self.execution
        output = context._empty((1,), np.float32)
        self._launch(
            "scalar_leaf",
            1,
            context._array(hist.total),
            g,
            h,
            regularization,
            context._array(output),
        )
        self._validate(
            context._array(output).reshape(1, 1),
            message="finite leaf with nonnegative curvature and positive finite denominator required",
        )
        return output

    def release(self, record):
        """Release a record's owned allocations; caller-supplied buffers stay caller-owned."""
        self.execution._check()
        if record not in self._records:
            raise ValueError("foreign, forged or released device record")
        for handle in self._records[record][1]:
            if handle in self.execution._buffers:
                self.execution.release(handle)
        del self._records[record]
