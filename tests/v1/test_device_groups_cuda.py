"""Real-CUDA contracts for grouped reductions; collection is not execution."""

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openboost import NumericData, Problem, device_inputs
from openboost.binning import Binning
from openboost.device import DeviceOperations, _workspace
from openboost.device_groups import HistogramJob, histogram_plan, histograms
from openboost.execution import ExecutionContext

from .reference.grouped_histogram import finite, histogram

pytestmark = pytest.mark.gpu


def setup(ops, count=3, width=2, *, empty=False):
    data = NumericData([[0, 1], [0, 3], [1, np.nan], [1, 4], [np.nan, 2], [0, 5], [1, 1]],
                       np.arange(7), ('x', 'y'))
    cuts = Binning.fit(data, bins=5)
    binned = cuts.transform(data)
    features = device_inputs.prepare(ops, binned)
    jobs, expected = [], {}
    for i in range(count):
        p = Problem(data, (np.arange(7) + i)[:, None], data.row_ids,
                    weight=np.arange(7) + i + 1.)
        resident = device_inputs.bind(ops, features, p)
        values = np.array([[((r + 3) * (i + 2) % 11 - 5) * (q + 1) / 8
                            for q in range(width)] for r in range(7)], np.float32)
        if i % 3 == 0:
            values[:3, 0] = [2**24, 1, -2**24]
        fields = ops.fields(resident, ops.execution.upload(values),
                            names=tuple(f'q{q}' for q in range(width)), roles=('training',) * width)
        rows = [] if empty else np.roll(np.arange(7)[::-1], i)[:(i % 7) + 1].tolist()
        job = HistogramJob(f'run-{i}', resident, fields, ops.rows(resident, rows))
        jobs.append(job)
        expected[job.run_id] = histogram(binned.codes, binned.missing, cuts.bin_counts, values, rows)
    return tuple(jobs), expected, binned


def exported(context, result):
    return tuple(context.export(h) for h in (result.sums, result.counts, result.total))


def check_result(context, actual, expected):
    assert actual.status == 'complete' and actual.error_type is None
    for got, wanted in zip(exported(context, actual.histogram), expected, strict=True):
        np.testing.assert_array_equal(got, wanted)


def save(tmp_path, name, record):
    root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'grouped-histograms'
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('width', [1, 2, 5])
def test_grouped_original_reversed_regrouped_reductions_and_owned_outputs(count, width, monkeypatch, tmp_path):
    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, expected, binned = setup(ops, count, width)
            # The old independent resident operation is also an exact control.
            for job in jobs:
                single = ops.histogram(job.data, job.fields, job.rows)
                for a, b in zip(exported(context, single), expected[job.run_id], strict=True):
                    np.testing.assert_array_equal(a, b)
                ops.release(single)
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            snapshots, schedules = [], (jobs, jobs[::-1], jobs[::2], jobs[1::2])
            original_export = context.export
            for schedule in filter(None, schedules):
                before = dict(context.metrics)
                size = len(schedule)

                def compact_only(handle, size=size):
                    assert handle.shape == (size,) and handle.dtype == np.dtype('i4').str
                    return original_export(handle)

                with monkeypatch.context() as patch:
                    patch.setattr(context, 'export', compact_only)
                    patch.setattr(ops, 'histogram', lambda *args: pytest.fail('sequential dispatch is not grouping'))
                    results = histograms(ops, schedule)
                after = dict(context.metrics)
                delta = {k: after[k] - before.get(k, 0) for k in after}
                assert delta['grouped_histogram_kernel_launches'] == 2
                assert delta['grouped_validation_kernel_launches'] == 1
                assert delta['kernel_launches'] == 3
                assert delta['upload_bytes'] == 5 * len(schedule)
                assert delta['export_bytes'] == 4 * len(schedule)
                assert delta['grouped_pack_bytes'] == sum(j.fields.values.nbytes + j.rows.positions.nbytes for j in schedule)
                assert delta['grouped_unpack_bytes'] == histogram_plan(schedule).detached_bytes
                assert delta['device_copy_bytes'] == delta['grouped_pack_bytes'] + delta['grouped_unpack_bytes']
                assert tuple(r.run_id for r in results) == tuple(j.run_id for j in schedule)
                observed = []
                for actual in results:
                    check_result(context, actual, expected[actual.run_id])
                    a, c, t = exported(context, actual.histogram)
                    observed.append(dict(run_id=actual.run_id, sums=a.tolist(), counts=c.tolist(), total=t.tolist()))
                    ops.release(actual.histogram)
                assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
                snapshots.append(dict(run_ids=[j.run_id for j in schedule], metrics=delta, observed=observed))
            save(tmp_path, f'M{count}-Q{width}.json', dict(
                count=count, width=width, scope='grouped operation only; no train-many or cost verdict',
                codes=binned.codes.tolist(), missing=binned.missing.tolist(), bins=list(binned.binning.bin_counts),
                jobs=[dict(run_id=j.run_id, fields=context.export(j.fields.values).tolist(),
                           rows=context.export(j.rows.positions).tolist(),
                           sums=expected[j.run_id][0].tolist(), counts=expected[j.run_id][1].tolist(),
                           total=expected[j.run_id][2].tolist()) for j in jobs], schedules=snapshots))
        assert context.metrics['live_bytes'] == 0 and not ops._records


@pytest.mark.parametrize('empty', [False, True])
def test_inactive_slots_skip_inputs_and_all_inactive_launches_nothing(empty):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, expected, _ = setup(ops, 3, empty=empty)
            before = dict(context.metrics)
            result = histograms(ops, jobs, active=(False,) * 3)
            assert [r.status for r in result] == ['inactive'] * 3
            assert dict(context.metrics) == before
            result = histograms(ops, jobs, active=(True, False, True))
            assert result[1].status == 'inactive' and result[1].histogram is None
            assert context.metrics['grouped_histogram_slots'] == 2
            for i in (0, 2):
                check_result(context, result[i], expected[result[i].run_id])
            assert context.metrics['grouped_pack_bytes'] == sum(j.fields.values.nbytes + j.rows.positions.nbytes for j in (jobs[0], jobs[2]))


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('kind', ['bucket', 'total'])
def test_numerical_failure_preserves_neighbors_and_same_id_retry(count, kind, tmp_path):
    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, expected, binned = setup(ops, count)
            bad = count // 2
            job = jobs[bad]
            values = np.zeros((7, 2), np.float32)
            # Feature x has rows 0/1 in bucket zero, 2/3 in bucket one.
            values[:4, 0] = ([3e38, 3e38, -3e38, -3e38] if kind == 'bucket'
                             else [2e38, -1e38, 2e38, -1e38])
            fields = ops.fields(job.data, context.upload(values), names=job.fields.names, roles=job.fields.roles)
            broken = replace(job, fields=fields, rows=ops.rows(job.data, [0, 2, 1, 3]))
            assert not finite(histogram(binned.codes, binned.missing, binned.binning.bin_counts, values, [0, 2, 1, 3]))
            changed = (*jobs[:bad], broken, *jobs[bad + 1:])
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            results = histograms(ops, changed)
            failure_metrics = dict(context.metrics)
            assert results[bad].status == 'failed' and results[bad].histogram is None
            assert results[bad].error_type == 'ValueError'
            observed = []
            for i, result in enumerate(results):
                arrays = None if result.histogram is None else {
                    key: value.tolist() for key, value in zip(
                        ('sums', 'counts', 'total'), exported(context, result.histogram), strict=True)}
                observed.append(dict(run_id=result.run_id, status=result.status, error_type=result.error_type,
                                     error_message=result.error_message, result=arrays))
                if i != bad:
                    check_result(context, result, expected[result.run_id])
                    ops.release(result.histogram)
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
            retry = histograms(ops, (job,))[0]
            check_result(context, retry, expected[job.run_id])
            retry_arrays = {key: value.tolist() for key, value in zip(
                ('sums', 'counts', 'total'), exported(context, retry.histogram), strict=True)}
            ops.release(retry.histogram)
            assert context.metrics['live_bytes'] == baseline
            save(tmp_path, f'failure-M{count}-{kind}.json', dict(
                count=count, kind=kind, failed_id=job.run_id, scope='grouped numerical failure and same-ID retry',
                codes=binned.codes.tolist(), missing=binned.missing.tolist(), bins=list(binned.binning.bin_counts),
                jobs=[dict(run_id=j.run_id, fields=context.export(j.fields.values).tolist(),
                           rows=context.export(j.rows.positions).tolist()) for j in changed],
                outcomes=observed, failure_metrics=failure_metrics, final_metrics=dict(context.metrics),
                retry=dict(run_id=job.run_id, fields=context.export(job.fields.values).tolist(),
                           rows=context.export(job.rows.positions).tolist(), result=retry_arrays),
                inputs_live_bytes=baseline, live_bytes_after_release=context.metrics['live_bytes']))


@pytest.mark.parametrize('fault', ['forged_fields', 'released_fields', 'released_rows', 'released_data', 'foreign_ops'])
def test_live_registration_is_checked_before_group_dispatch(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _ = setup(ops)
            job = jobs[1]
            if fault == 'forged_fields':
                jobs = (jobs[0], replace(job, fields=replace(job.fields)), jobs[2])
            elif fault.startswith('released'):
                ops.release(getattr(job, fault.removeprefix('released_')))
            active_ops = DeviceOperations(context) if fault == 'foreign_ops' else ops
            before = dict(context.metrics)
            with pytest.raises(ValueError, match='foreign|released|forged'):
                histograms(active_ops, jobs, active=(True, False, True))
            assert context.metrics['live_bytes'] == before['live_bytes']
            assert context.metrics['kernel_launches'] == before['kernel_launches']


@pytest.mark.parametrize('stage', [1, 4, 8, 10])
def test_partial_allocation_failure_is_atomic_and_retry_keeps_inputs(stage, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, expected, _ = setup(ops)
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            original, calls = context._empty, 0

            def fail(shape, dtype):
                nonlocal calls
                calls += 1
                if calls == stage:
                    raise MemoryError('declared allocation fault')
                return original(shape, dtype)

            with monkeypatch.context() as patch:
                patch.setattr(context, '_empty', fail)
                with pytest.raises(MemoryError, match='declared allocation fault'):
                    histograms(ops, jobs)
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
            for result in histograms(ops, jobs):
                check_result(context, result, expected[result.run_id])
                ops.release(result.histogram)
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records


def test_physical_pool_cap_failure_discards_group_scratch_then_retry_succeeds():
    cap = 1024**2
    with ExecutionContext(max_bytes=cap) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, expected, _ = setup(ops)
            scratch = context.upload(np.zeros((cap - context.metrics['peak_pool_bytes']) // 4, np.float32))
            assert context.metrics['peak_pool_bytes'] == cap
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            with pytest.raises(MemoryError):
                histograms(ops, jobs)
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
            context.release(scratch)
            for result in histograms(ops, jobs):
                check_result(context, result, expected[result.run_id])
                ops.release(result.histogram)
            assert context.metrics['peak_pool_bytes'] <= cap


@pytest.mark.parametrize('vector', [False, True])
def test_group_outputs_compose_with_existing_candidates_leaves_and_routing(vector):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _ = setup(ops, 3)
            prepared = []
            for i, job in enumerate(jobs):
                g = (np.arange(7) - 3 - i).astype(np.float32)
                values = np.column_stack((g, np.ones(7, np.float32)))
                names = ('gradient', 'curvature')
                if vector:
                    values = np.column_stack((values, g / 2, np.full(7, 2, np.float32)))
                    names = ('gradient:0', 'curvature:0', 'gradient:1', 'curvature:1')
                fields = ops.fields(job.data, context.upload(values), names=names, roles=('training',) * len(names))
                prepared.append(replace(job, fields=fields, rows=ops.rows(job.data)))
            results = histograms(ops, tuple(prepared))
            score = ops.vector_scores if vector else ops.newton_scores
            feasible = ops.vector_feasible if vector else ops.feasible
            leaf = ops.vector_leaf if vector else ops.leaf

            def downstream(hist):
                candidates = ops.candidates(hist)
                scores = score(candidates)
                mask = feasible(candidates)
                split = ops.choose(candidates, scores, mask)
                children = None if split is None else tuple(
                    context.export(r.positions).tolist() for r in ops.partition(hist.rows, split))
                return (context.export(candidates.values), context.export(scores.values),
                        context.export(mask.values), context.export(leaf(hist)),
                        None if split is None else split.key, children)

            for job, outcome in zip(prepared, results, strict=True):
                single = ops.histogram(job.data, job.fields, job.rows)
                a, b = downstream(single), downstream(outcome.histogram)
                for x, y in zip(a[:4], b[:4], strict=True):
                    np.testing.assert_array_equal(x, y)
                assert a[4:] == b[4:]
                ops.release(outcome.histogram)


def test_separately_prepared_feature_handles_and_foreign_context_fail_explicitly():
    with ExecutionContext() as first, ExecutionContext() as second:
        ops, other = DeviceOperations(first), DeviceOperations(second)
        with _workspace(ops), _workspace(other):
            jobs, _, _ = setup(ops)
            independent, _, _ = setup(ops)
            foreign, _, _ = setup(other)
            for changed in (independent[1], foreign[1]):
                with pytest.raises(ValueError, match='shared feature handles'):
                    histograms(ops, (jobs[0], changed, jobs[2]))
            with pytest.raises(ValueError, match='foreign|forged|released'):
                histograms(ops, foreign)
