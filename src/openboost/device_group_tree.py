"""Experimental grouped stable partitions and provenance-bearing tree predictions."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .device import DeviceData, DeviceRows, DeviceSplit, _atomic, _workspace
from .device_groups import _buffer, _copy, _count
from .device_tree import DeviceTree, _check_binning
from .execution import DeviceBuffer


@dataclass(frozen=True)
class PartitionJob:
    run_id: str
    rows: DeviceRows
    split: DeviceSplit


@dataclass(frozen=True)
class PartitionOutcome:
    run_id: str
    status: str
    children: tuple[DeviceRows, DeviceRows] | None = None


@dataclass(frozen=True)
class PredictionJob:
    run_id: str
    tree: DeviceTree
    data: DeviceData


@dataclass(frozen=True, eq=False)
class DevicePrediction:
    """Registered snapshot bound to its exact source; owns values, not tree/data.

    Release through ops.release. Data must remain live for record operations.
    Releasing the tree does not release values; consumers needing a live tree
    must check it separately. Constructing this dataclass does not register it.
    """

    tree: DeviceTree
    data: DeviceData
    values: DeviceBuffer


@dataclass(frozen=True)
class PredictionOutcome:
    run_id: str
    status: str
    prediction: DevicePrediction | None = None
    error_type: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class PartitionPlan:
    run_ids: tuple[str, ...]
    active: tuple[bool, ...]
    row_counts: tuple[int, ...]
    keys: tuple[tuple[int, int, bool], ...]

    @property
    def shape(self):
        return len(self.run_ids), 2, max(self.row_counts)

    @property
    def packed_bytes(self):
        return len(self.run_ids) * (max(self.row_counts) * 4 + 17)

    @property
    def output_bytes(self):
        return len(self.run_ids) * (2 * max(self.row_counts) * 4 + 8)

    @property
    def detached_bytes(self):
        return sum(n * 4 for n, active in zip(self.row_counts, self.active, strict=True) if active)

    @property
    def peak_extra_bytes(self):
        return self.packed_bytes + self.output_bytes + self.detached_bytes if any(self.active) else 0


@dataclass(frozen=True)
class PredictionPlan:
    run_ids: tuple[str, ...]
    active: tuple[bool, ...]
    node_counts: tuple[int, ...]
    shape: tuple[int, int, int]

    @property
    def packed_bytes(self):
        m, _, width = self.shape
        return m * (max(self.node_counts) * (20 + 4 * width) + 1)

    @property
    def output_bytes(self):
        m, n, width = self.shape
        return m * (n * width * 4 + 4)

    @property
    def detached_bytes(self):
        _, n, width = self.shape
        return sum(self.active) * n * width * 4

    @property
    def peak_extra_bytes(self):
        return self.packed_bytes + self.output_bytes + self.detached_bytes if any(self.active) else 0


def _jobs(jobs, active, kind):
    if not isinstance(jobs, tuple) or not 1 <= len(jobs) <= 32:
        raise ValueError('explicit tuple of 1–32 jobs required')
    if active is None:
        active = (True,) * len(jobs)
    if not isinstance(active, tuple) or len(active) != len(jobs) or any(type(a) is not bool for a in active):
        raise ValueError('explicit aligned tuple of boolean active flags required')
    ids = []
    for job in jobs:
        if not isinstance(job, kind) or not isinstance(job.run_id, str) or not job.run_id:
            raise ValueError('named operation job required')
        if job.run_id in ids:
            raise ValueError('distinct run IDs required')
        ids.append(job.run_id)
    return tuple(ids), active


def _features(data):
    if not isinstance(data, DeviceData) or type(data.n_rows) is not int or not 0 < data.n_rows <= np.iinfo(np.int32).max:
        raise ValueError('DeviceData with positive int32 row capacity required')
    if not data.bin_counts or len(data.bin_counts) != len(data.feature_names) or any(
        type(b) is not int or not 1 <= b <= np.iinfo(np.int32).max for b in data.bin_counts
    ):
        raise ValueError('positive int32 feature bin counts required')
    _buffer(data.codes, (len(data.bin_counts), data.n_rows), 'i4')
    _buffer(data.missing, (len(data.bin_counts), data.n_rows), 'b1')
    return (data.data_identity, data.prepared_identity, data.binning_identity,
            data.n_rows, data.feature_names, data.bin_counts, data.codes, data.missing)


def partition_plan(jobs, *, active=None):
    """Host compatibility/byte bounds; execution also checks every live split chain."""
    ids, active = _jobs(jobs, active, PartitionJob)
    first, counts, keys = None, [], []
    for job in jobs:
        if not isinstance(job.rows, DeviceRows) or not isinstance(job.split, DeviceSplit):
            raise ValueError('DeviceRows and DeviceSplit required')
        identity = _features(job.rows.data)
        if first is not None and first != identity:
            raise ValueError('shared feature handles and preparation required')
        first = identity
        # Type/shape checks on the candidate chain use the existing public types;
        # registered execution below rejects forged or released ancestors too.
        from .device import DeviceCandidates, DeviceHistogram

        batch = job.split.candidates
        if not isinstance(batch, DeviceCandidates) or not isinstance(batch.histogram, DeviceHistogram):
            raise ValueError('candidate histogram chain required')
        if batch.histogram.rows is not job.rows or batch.data is not job.rows.data:
            raise ValueError('split belongs to different routed rows')
        row_buffer = job.rows.positions
        if not isinstance(row_buffer, DeviceBuffer) or len(row_buffer.shape) != 1:
            raise ValueError('one-dimensional row buffer required')
        n = row_buffer.shape[0]
        if type(n) is not int or not 0 <= n <= job.rows.data.n_rows:
            raise ValueError('row capacity exceeded')
        _buffer(row_buffer, (n,), 'i4')
        key = job.split.key
        if not 0 <= key[0] < len(job.rows.data.bin_counts) or not 0 <= key[1] < job.rows.data.bin_counts[key[0]]:
            raise ValueError('in-range feature threshold required')
        counts.append(n)
        keys.append(key)
    return PartitionPlan(ids, active, tuple(counts), tuple(keys))


def prediction_plan(jobs, *, active=None):
    """Common feature handles/output width; tree topology and values stay distinct."""
    ids, active = _jobs(jobs, active, PredictionJob)
    first, counts, width = None, [], None
    for job in jobs:
        identity = _features(job.data)
        if first is not None and first != identity:
            raise ValueError('shared feature handles and preparation required')
        first = identity
        if not isinstance(job.tree, DeviceTree):
            raise ValueError('DeviceTree required')
        _check_binning(job.data, job.tree.binning)
        if not 0 < job.tree.n_nodes <= np.iinfo(np.int32).max:
            raise ValueError('positive int32 node count required')
        if job.tree.output_width > np.iinfo(np.int32).max:
            raise ValueError('int32 output width required')
        if width is not None and width != job.tree.output_width:
            raise ValueError('common tree output width required')
        width = job.tree.output_width
        counts.append(job.tree.n_nodes)
    return PredictionPlan(ids, active, tuple(counts), (len(jobs), jobs[0].data.n_rows, width))


def _launch(ops, name, size, *args):
    from numba import cuda

    from . import _device_group_tree_kernels

    context = ops.execution
    start = perf_counter()
    with context._scope():
        stream = cuda.external_stream(context._stream.ptr)
        arrays = [cuda.as_cuda_array(a, sync=False) if isinstance(a, context._cp.ndarray) else a for a in args]
        context._counts['kernel_launches'] += 1
        try:
            getattr(_device_group_tree_kernels, name)[max(1, (size + 127) // 128), 128, stream](*arrays)
        finally:
            context._counts['kernel_dispatch_seconds'] += perf_counter() - start


@_atomic
def partitions(ops, jobs, *, active=None):
    """One M-wide stable partition; only 2M int32 child lengths leave the device."""
    plan = partition_plan(jobs, active=active)
    context = ops.execution
    for job in jobs:
        ops._get(job.rows, DeviceRows)
        ops._get(job.split, DeviceSplit)
        ops._batch(job.split.candidates)
    if not any(plan.active):
        return tuple(PartitionOutcome(j.run_id, 'inactive') for j in jobs)
    context._reserve(plan.peak_extra_bytes)
    _count(context, 'grouped_partition_calls', 1)
    m, _, capacity = plan.shape
    with _workspace(ops) as retained:
        rows = context._empty((m, capacity), np.int32)
        lengths = context.upload(np.asarray(plan.row_counts, np.int32))
        keys = context.upload(np.asarray(plan.keys, np.int32))
        mask = context.upload(np.asarray(plan.active, bool))
        _count(context, 'grouped_partition_metadata_upload_bytes', 17 * m)
        for i, job in enumerate(jobs):
            if plan.active[i]:
                _copy(context, context._array(rows)[i, :plan.row_counts[i]],
                      context._array(job.rows.positions), 'grouped_partition_pack_bytes')
        output = context._empty(plan.shape, np.int32)
        sizes = context._empty((m, 2), np.int32)
        data = jobs[0].rows.data
        _launch(ops, 'partitions', m, context._array(data.codes), context._array(data.missing),
                context._array(rows), context._array(lengths), context._array(keys),
                context._array(mask), context._array(output), context._array(sizes))
        _count(context, 'grouped_partition_kernel_launches', 1)
        _count(context, 'grouped_partition_slots', sum(plan.active))
        observed = context.export(sizes)
        _count(context, 'grouped_partition_export_bytes', sizes.nbytes)
        results = []
        for i, job in enumerate(jobs):
            a, b = map(int, observed[i])
            expected = plan.row_counts[i] if plan.active[i] else 0
            if min(a, b) < 0 or a + b != expected:
                raise RuntimeError('invalid grouped device child sizes')
            if not plan.active[i]:
                results.append(PartitionOutcome(job.run_id, 'inactive'))
                continue
            children = []
            for side, n in enumerate((a, b)):
                handle = context._empty((n,), np.int32)
                _copy(context, context._array(handle), context._array(output)[i, side, :n],
                      'grouped_partition_unpack_bytes')
                record = ops._record(DeviceRows(job.rows.data, handle), (handle,), (handle,))
                children.append(record)
                retained.add(record)
            results.append(PartitionOutcome(job.run_id, 'complete', tuple(children)))
        return tuple(results)


@_atomic
def predictions(ops, jobs, *, active=None):
    """M-wide prediction and finite check; detached values retain source provenance."""
    plan = prediction_plan(jobs, active=active)
    context = ops.execution
    for job in jobs:
        ops._get(job.data, DeviceData)
        ops._get(job.tree, DeviceTree)
        topology, values = ops._records[job.tree][0]
        _buffer(topology, (job.tree.n_nodes, 5), 'i4')
        value_shape = (job.tree.n_nodes,) if job.tree.output_width == 1 else (job.tree.n_nodes, job.tree.output_width)
        _buffer(values, value_shape, 'f4')
    if not any(plan.active):
        return tuple(PredictionOutcome(j.run_id, 'inactive') for j in jobs)
    context._reserve(plan.peak_extra_bytes)
    _count(context, 'grouped_prediction_calls', 1)
    m, n, width = plan.shape
    with _workspace(ops) as retained:
        topology = context._empty((m, max(plan.node_counts), 5), np.int32)
        values = context._empty((m, max(plan.node_counts), width), np.float32)
        mask = context.upload(np.asarray(plan.active, bool))
        _count(context, 'grouped_prediction_metadata_upload_bytes', m)
        for i, job in enumerate(jobs):
            if plan.active[i]:
                t, v = ops._records[job.tree][0]
                _copy(context, context._array(topology)[i, :job.tree.n_nodes], context._array(t),
                      'grouped_prediction_pack_bytes')
                _copy(context, context._array(values)[i, :job.tree.n_nodes],
                      context._array(v).reshape(job.tree.n_nodes, width), 'grouped_prediction_pack_bytes')
        output = context._empty(plan.shape, np.float32)
        flags = context._empty((m,), np.int32)
        data = jobs[0].data
        _launch(ops, 'predictions', m * n, context._array(data.codes), context._array(data.missing),
                context._array(topology), context._array(values), context._array(mask), context._array(output))
        _count(context, 'grouped_prediction_kernel_launches', 1)
        _launch(ops, 'finite_predictions', m, context._array(output), context._array(flags))
        _count(context, 'grouped_prediction_validation_kernel_launches', 1)
        _count(context, 'grouped_prediction_slots', sum(plan.active))
        failed = context.export(flags)
        _count(context, 'grouped_prediction_export_bytes', flags.nbytes)
        results = []
        for i, job in enumerate(jobs):
            if not plan.active[i]:
                results.append(PredictionOutcome(job.run_id, 'inactive'))
            elif failed[i]:
                _count(context, 'grouped_prediction_failures', 1)
                results.append(PredictionOutcome(job.run_id, 'failed', error_type='ValueError',
                                                 error_message='finite tree predictions required'))
            else:
                handle = context._empty((n, width), np.float32)
                _copy(context, context._array(handle), context._array(output)[i], 'grouped_prediction_unpack_bytes')
                record = ops._record(DevicePrediction(job.tree, job.data, handle), (handle,), (handle,))
                retained.add(record)
                _count(context, 'grouped_prediction_successes', 1)
                results.append(PredictionOutcome(job.run_id, 'complete', record))
        return tuple(results)
