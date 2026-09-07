"""Experimental public CUDA named fields and routed histograms; no training yet."""

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
        for key in ("kernel_launches", "kernel_dispatch_seconds", "validation_export_bytes"):
            execution._counts.setdefault(key, 0)

    def _record(self, record, handles, owned=()):
        self._records[record] = (tuple(handles), tuple(owned))
        return record

    def _get(self, record, kind):
        self.execution._check()
        if not isinstance(record, kind) or record not in self._records:
            raise ValueError("foreign, forged or released device record")
        if not isinstance(record, DeviceData):
            self._get(record.data, DeviceData)
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

    def _validate(self, array, *, nonnegative=False):
        context = self.execution
        flags = context._empty((array.shape[1],), np.int32)
        self._launch("validate_fields", array.shape[1], array, nonnegative, context._array(flags))
        self._flags(
            flags, "finite fields required; declared nonnegative information cannot be negative"
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
            selected = (
                np.arange(data.n_rows, dtype=np.int32)
                if positions is None
                else np.asarray(positions)
            )
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

    def release(self, record):
        """Release a record's owned allocations; caller-supplied buffers stay caller-owned."""
        self.execution._check()
        if record not in self._records:
            raise ValueError("foreign, forged or released device record")
        for handle in self._records[record][1]:
            if handle in self.execution._buffers:
                self.execution.release(handle)
        del self._records[record]
