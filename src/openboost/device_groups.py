"""Experimental grouped resident operations with explicit compatible jobs."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .device import (
    DeviceData,
    DeviceFields,
    DeviceHistogram,
    DeviceRows,
    _atomic,
    _schema,
    _workspace,
)
from .execution import DeviceBuffer


@dataclass(frozen=True)
class HistogramJob:
    """One named original-row reduction; no objective state is shared implicitly."""

    run_id: str
    data: DeviceData
    fields: DeviceFields
    rows: DeviceRows


@dataclass(frozen=True)
class HistogramOutcome:
    """Complete, inactive or numerically failed slot in caller order.

    A complete histogram is independently owned and released with ops.release.
    Allocation/driver failures raise for the whole operation without outputs.
    """

    run_id: str
    status: str
    histogram: DeviceHistogram | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class HistogramPlan:
    """Host metadata only; byte counts exclude inputs and physical pool rounding."""

    run_ids: tuple[str, ...]
    active: tuple[bool, ...]
    row_counts: tuple[int, ...]
    n_rows: int
    shape: tuple[int, int, int, int]

    @property
    def packed_bytes(self):
        m, _, _, q = self.shape
        return m * (self.n_rows * q * 4 + max(self.row_counts) * 4 + 5)

    @property
    def _one_output_bytes(self):
        _, f, b, q = self.shape
        return f * b * q * 4 + f * b * 8 + q * 4

    @property
    def reduction_bytes(self):
        return len(self.run_ids) * (self._one_output_bytes + 4)

    @property
    def detached_bytes(self):
        return sum(self.active) * self._one_output_bytes

    @property
    def peak_extra_bytes(self):
        """Conservative logical bound; a physical pool cap can still reject it."""
        return self.packed_bytes + self.reduction_bytes + self.detached_bytes if any(self.active) else 0


def _buffer(handle, shape, dtype):
    if (
        not isinstance(handle, DeviceBuffer)
        or handle.shape != shape
        or handle.dtype != np.dtype(dtype).str
        or handle.nbytes != int(np.prod(shape)) * np.dtype(dtype).itemsize
    ):
        raise ValueError("aligned context buffer metadata required")


def histogram_plan(jobs, *, active=None):
    """Check metadata compatibility; execution additionally checks live registration.

    Supports 1–32 distinct run IDs with identical field layout and borrowed feature
    handles. Problem weights, field values and row selections remain independent.
    Inactive jobs must still be valid. The caller explicitly groups compatible work.
    """
    if not isinstance(jobs, tuple) or not 1 <= len(jobs) <= 32:
        raise ValueError("explicit tuple of 1–32 histogram jobs required")
    if active is None:
        active = (True,) * len(jobs)
    if not isinstance(active, tuple) or len(active) != len(jobs) or any(type(x) is not bool for x in active):
        raise ValueError("explicit aligned tuple of boolean active flags required")
    ids, counts = [], []
    first = None
    for job in jobs:
        if not isinstance(job, HistogramJob) or not isinstance(job.run_id, str) or not job.run_id:
            raise ValueError("named HistogramJob required")
        if job.run_id in ids:
            raise ValueError("distinct run IDs required")
        data, fields, rows = job.data, job.fields, job.rows
        if not isinstance(data, DeviceData) or not isinstance(fields, DeviceFields) or not isinstance(rows, DeviceRows):
            raise ValueError("DeviceData, DeviceFields and DeviceRows required")
        if fields.data is not data or rows.data is not data:
            raise ValueError("each job's fields and rows must share its exact data binding")
        if not isinstance(data.n_rows, int) or not 0 < data.n_rows <= np.iinfo(np.int32).max:
            raise ValueError("positive int32 row capacity required")
        names, roles = _schema(fields.names, fields.roles, len(fields.names))
        if "unweighted" in roles:
            raise ValueError("apply objective training weights before aggregation")
        if not data.bin_counts or len(data.bin_counts) != len(data.feature_names) or any(type(b) is not int or b < 1 for b in data.bin_counts):
            raise ValueError("positive feature bin counts required")
        _buffer(data.codes, (len(data.feature_names), data.n_rows), 'i4')
        _buffer(data.missing, (len(data.feature_names), data.n_rows), 'b1')
        _buffer(fields.values, (data.n_rows, len(names)), 'f4')
        if not isinstance(rows.positions, DeviceBuffer) or len(rows.positions.shape) != 1:
            raise ValueError("one-dimensional row buffer required")
        nr = rows.positions.shape[0]
        if not isinstance(nr, int) or not 0 <= nr <= data.n_rows:
            raise ValueError("row capacity exceeded")
        _buffer(rows.positions, (nr,), 'i4')
        identity = (data.data_identity, data.prepared_identity, data.binning_identity,
                    data.n_rows, data.feature_names, data.bin_counts, names, roles)
        if first is None:
            first = (identity, data.codes, data.missing)
        elif identity != first[0] or data.codes is not first[1] or data.missing is not first[2]:
            raise ValueError("shared feature handles, preparation and field layout required")
        ids.append(job.run_id)
        counts.append(nr)
    data, fields = jobs[0].data, jobs[0].fields
    return HistogramPlan(tuple(ids), active, tuple(counts), data.n_rows,
                         (len(jobs), len(data.bin_counts), max(data.bin_counts) + 1, len(fields.names)))


def _launch(ops, name, size, *args):
    # Separate module leaves every consumed scalar/vector kernel unchanged.
    from numba import cuda

    from . import _device_group_kernels

    context = ops.execution
    start = perf_counter()
    with context._scope():
        stream = cuda.external_stream(context._stream.ptr)
        arrays = [cuda.as_cuda_array(a, sync=False) if isinstance(a, context._cp.ndarray) else a for a in args]
        context._counts['kernel_launches'] += 1
        try:
            getattr(_device_group_kernels, name)[max(1, (size + 127) // 128), 128, stream](*arrays)
        finally:
            context._counts['kernel_dispatch_seconds'] += perf_counter() - start


def _count(context, name, amount):
    context._counts[name] = context._counts.get(name, 0) + amount


def _copy(context, destination, source, counter):
    context._cp.copyto(destination, source)
    _count(context, counter, source.nbytes)
    _count(context, 'device_copy_bytes', source.nbytes)


@_atomic
def histograms(ops, jobs, *, active=None):
    """Execute two grouped reductions, then one grouped per-slot finite check.

    Fields and row buffers stay on device; only M int32 validation flags return.
    Packing preserves each input row sequence and float32 addition order. Outputs
    are ordinary independently owned DeviceHistograms, compatible with public
    candidate/leaf operations. This is an operation, not a train-many scheduler.
    """
    plan = histogram_plan(jobs, active=active)
    context = ops.execution
    for job in jobs:
        ops._get(job.data, DeviceData)
        ops._get(job.fields, DeviceFields)
        ops._get(job.rows, DeviceRows)
        ops._float(job.fields.values, (plan.n_rows, plan.shape[-1]))
    if not any(plan.active):
        return tuple(HistogramOutcome(j.run_id, 'inactive') for j in jobs)
    context._reserve(plan.peak_extra_bytes)
    _count(context, 'grouped_histogram_calls', 1)
    m, f, b, q = plan.shape
    with _workspace(ops) as retained:
        values = context._empty((m, plan.n_rows, q), np.float32)
        rows = context._empty((m, max(plan.row_counts)), np.int32)
        lengths = context.upload(np.array(plan.row_counts, np.int32))
        mask = context.upload(np.array(plan.active, bool))
        _count(context, 'grouped_metadata_upload_bytes', lengths.nbytes + mask.nbytes)
        for i, job in enumerate(jobs):
            if plan.active[i]:
                _copy(context, context._array(values)[i], context._array(job.fields.values), 'grouped_pack_bytes')
                _copy(context, context._array(rows)[i, :plan.row_counts[i]],
                      context._array(job.rows.positions), 'grouped_pack_bytes')
        sums = context._empty(plan.shape, np.float32)
        counts = context._empty((m, f, b), np.int64)
        total = context._empty((m, q), np.float32)
        flags = context._empty((m,), np.int32)
        shared = jobs[0].data
        _launch(ops, 'histograms', m * f * b * q, context._array(shared.codes),
                context._array(shared.missing), shared.bin_counts, context._array(values),
                context._array(rows), context._array(lengths), context._array(mask),
                context._array(sums), context._array(counts))
        _count(context, 'grouped_histogram_kernel_launches', 1)
        _launch(ops, 'row_totals', m * q, context._array(values), context._array(rows),
                context._array(lengths), context._array(mask), context._array(total))
        _count(context, 'grouped_histogram_kernel_launches', 1)
        _count(context, 'grouped_histogram_slots', sum(plan.active))
        _launch(ops, 'finite_histograms', m, context._array(sums), context._array(total),
                context._array(flags))
        _count(context, 'grouped_validation_kernel_launches', 1)
        failed = context.export(flags)
        _count(context, 'validation_export_bytes', failed.nbytes)
        _count(context, 'grouped_validation_export_bytes', failed.nbytes)
        outcomes = []
        for i, job in enumerate(jobs):
            if not plan.active[i]:
                outcomes.append(HistogramOutcome(job.run_id, 'inactive'))
            elif failed[i]:
                _count(context, 'grouped_histogram_failures', 1)
                outcomes.append(HistogramOutcome(job.run_id, 'failed', error_type='ValueError',
                                                 error_message='finite histogram sums and row totals required'))
            else:
                handles = []
                for packed in (sums, counts, total):
                    source = context._array(packed)[i]
                    handle = context._empty(tuple(source.shape), source.dtype)
                    _copy(context, context._array(handle), source, 'grouped_unpack_bytes')
                    handles.append(handle)
                histogram = ops._record(DeviceHistogram(job.data, job.fields, job.rows, *handles), handles, handles)
                retained.add(histogram)
                _count(context, 'grouped_histogram_successes', 1)
                outcomes.append(HistogramOutcome(job.run_id, 'complete', histogram))
        return tuple(outcomes)
