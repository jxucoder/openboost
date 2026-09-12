"""134 grouped routing/prediction contracts; execute only on real CUDA hardware."""

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openboost import NumericData, Problem, device_inputs, device_tree
from openboost.binning import Binning
from openboost.device import DeviceOperations, _workspace
from openboost.device_group_tree import (
    DevicePrediction,
    PartitionJob,
    PredictionJob,
    partition_plan,
    partitions,
    prediction_plan,
    predictions,
)
from openboost.device_groups import HistogramJob, histograms
from openboost.execution import ExecutionContext

from .grouped_tree_artifacts import FRESH, snapshot
from .multi_squared_artifacts import fresh_command
from .reference import grouped_tree as ref

pytestmark = pytest.mark.gpu


def setup(ops, count=3, width=2, *, empty=False, make_trees=True):
    x = np.array([[0, 1], [0, 3], [1, np.nan], [1, 4], [np.nan, 2], [0, 5], [1, 1]])
    data = NumericData(x, np.arange(7), ('x', 'y'))
    binned = Binning.fit(data, bins=5).transform(data)
    features = device_inputs.prepare(ops, binned)
    routing, prediction, fields = [], [], []
    for i in range(count):
        problem = Problem(data, (np.arange(7) + i)[:, None], data.row_ids,
                          weight=np.arange(7) + i + 1.)
        bound = device_inputs.bind(ops, features, problem)
        values = np.empty((7, 2 * width), np.float32)
        for q in range(width):
            values[:, 2 * q] = [(((r + 1) * (q + 2) % (i % 3 + 5)) - 2) * (i + 1) / 8 for r in range(7)]
            values[:, 2 * q + 1] = np.arange(7) % 3 + 1
        field = ops.fields(bound, ops.execution.upload(values),
                           names=tuple(n for q in range(width) for n in (f'gradient:{q}', f'curvature:{q}')),
                           roles=('training',) * (2 * width))
        fields.append(field)
        selected = [] if empty else np.roll(np.arange(7)[::-1], i)[:(i % 7) + 1].tolist()
        rows = ops.rows(bound, selected)
        hist = ops.histogram(bound, field, rows)
        batch = ops.candidates(hist)
        # Explicit scores select a valid feature threshold even for empty rows.
        # Partitioning itself permits empty children, as the ordinary operation does.
        f, t, missing_left = i % 2, 0, bool(i % 3)
        scores = np.zeros(batch.size, np.float32)
        scores[next(k for k in range(batch.size) if batch.key(k) == (f, t, missing_left))] = 1
        split = ops.choose(batch, ops.scores(batch, ops.execution.upload(scores)),
                           ops.mask(batch, ops.execution.upload(np.ones(batch.size, bool))))
        assert split is not None
        routing.append(PartitionJob(f'run-{i}', rows, split))
        if make_trees:
            tree = device_tree.depthwise(
                ops, bound, field, binning=binned.binning, max_depth=i % 3, output_width=width,
                scoring=lambda o, b: o.vector_scores(b, reg_lambda=2),
                legality=lambda o, b: o.vector_feasible(b),
                leaf=lambda o, h: o.vector_leaf(h, reg_lambda=2),
            )
            prediction.append(PredictionJob(f'run-{i}', tree, bound))
    return tuple(routing), tuple(prediction), tuple(fields), binned


def save(tmp_path, name, record):
    root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'grouped-tree'
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return path


def delta(before, after):
    return {k: after[k] - before.get(k, 0) for k in after}


def release(ops, outcomes, operation):
    for outcome in outcomes:
        if outcome.status == 'complete':
            records = outcome.children if operation == 'partition' else (outcome.prediction,)
            for record in records:
                ops.release(record)


def expected_routes(context, jobs, binned):
    return {j.run_id: ref.partition(binned.codes, binned.missing, context.export(j.rows.positions).tolist(), j.split.key)
            for j in jobs}


def expected_predictions(ops, jobs, binned):
    return {j.run_id: ref.predict(binned.codes, binned.missing, j.tree.topology, device_tree.export(ops, j.tree).value)
            for j in jobs}


def check(context, outcomes, expected, operation):
    observed = []
    for outcome in outcomes:
        assert outcome.status == 'complete'
        if operation == 'partition':
            actual = [context.export(c.positions).tolist() for c in outcome.children]
            assert actual == list(expected[outcome.run_id])
        else:
            actual = context.export(outcome.prediction.values)
            np.testing.assert_array_equal(actual, expected[outcome.run_id])
            actual = actual.tolist()
        observed.append(dict(run_id=outcome.run_id, result=actual))
    return observed


@pytest.mark.parametrize('count', [1, 8, 32])
def test_grouped_original_reversed_regrouped_stable_partitions(count, monkeypatch, tmp_path):
    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _, binned = setup(ops, count, make_trees=False)
            expected = expected_routes(context, jobs, binned)
            for job in jobs:
                children = ops.partition(job.rows, job.split)
                assert [context.export(c.positions).tolist() for c in children] == list(expected[job.run_id])
                for child in children:
                    ops.release(child)
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            schedules = []
            original_export = context.export
            for group in filter(None, (jobs, jobs[::-1], jobs[::2], jobs[1::2])):
                before = dict(context.metrics)
                size = len(group)

                def compact(handle, size=size):
                    assert handle.shape == (size, 2) and handle.dtype == np.dtype('i4').str
                    return original_export(handle)

                with monkeypatch.context() as patch:
                    patch.setattr(context, 'export', compact)
                    patch.setattr(ops, 'partition', lambda *args: pytest.fail('sequential routing is not grouping'))
                    outcomes = partitions(ops, group)
                metrics = delta(before, context.metrics)
                assert metrics['kernel_launches'] == metrics['grouped_partition_kernel_launches'] == 1
                assert metrics['upload_bytes'] == metrics['grouped_partition_metadata_upload_bytes'] == 17 * len(group)
                assert metrics['export_bytes'] == metrics['grouped_partition_export_bytes'] == 8 * len(group)
                assert metrics['grouped_partition_pack_bytes'] == sum(j.rows.positions.nbytes for j in group)
                assert metrics['grouped_partition_unpack_bytes'] == partition_plan(group).detached_bytes
                assert metrics['device_copy_bytes'] == 2 * metrics['grouped_partition_pack_bytes']
                assert tuple(o.run_id for o in outcomes) == tuple(j.run_id for j in group)
                actual = check(context, outcomes, expected, 'partition')
                # Each output is a live ordinary row record usable by old consumers.
                for job, outcome in zip(group, outcomes, strict=True):
                    for child in outcome.children:
                        hist = ops.histogram(child.data, job.split.candidates.histogram.fields, child)
                        assert sum(context.export(hist.counts)[0]) == child.positions.shape[0]
                        ops.release(hist)
                release(ops, outcomes, 'partition')
                assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
                schedules.append(dict(run_ids=[j.run_id for j in group], metrics=metrics, observed=actual))
            save(tmp_path, f'partition-M{count}.json', dict(
                count=count, codes=binned.codes.tolist(), missing=binned.missing.tolist(),
                jobs=[dict(run_id=j.run_id, rows=context.export(j.rows.positions).tolist(), key=j.split.key) for j in jobs],
                schedules=schedules, scope='grouped stable routing only'))
        assert context.metrics['live_bytes'] == 0 and not ops._records


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('width', [1, 2, 3])
def test_grouped_variable_tree_original_reversed_regrouped_predictions(count, width, monkeypatch, tmp_path):
    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            _, jobs, _, binned = setup(ops, count, width)
            expected = expected_predictions(ops, jobs, binned)
            if count > 1:
                assert len({j.tree.n_nodes for j in jobs}) > 1
            for job in jobs:
                single = device_tree.predict(ops, job.tree, job.data)
                np.testing.assert_array_equal(context.export(single), expected[job.run_id])
                context.release(single)
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            schedules, original_export = [], context.export
            for group in filter(None, (jobs, jobs[::-1], jobs[::2], jobs[1::2])):
                before = dict(context.metrics)
                size = len(group)

                def compact(handle, size=size):
                    assert handle.shape == (size,) and handle.dtype == np.dtype('i4').str
                    return original_export(handle)

                with monkeypatch.context() as patch:
                    patch.setattr(context, 'export', compact)
                    patch.setattr(device_tree, 'predict', lambda *args: pytest.fail('sequential prediction is not grouping'))
                    outcomes = predictions(ops, group)
                metrics = delta(before, context.metrics)
                assert metrics['kernel_launches'] == 2
                assert metrics['grouped_prediction_kernel_launches'] == metrics['grouped_prediction_validation_kernel_launches'] == 1
                assert metrics['upload_bytes'] == metrics['grouped_prediction_metadata_upload_bytes'] == len(group)
                assert metrics['export_bytes'] == metrics['grouped_prediction_export_bytes'] == 4 * len(group)
                assert metrics['grouped_prediction_pack_bytes'] == sum(j.tree.n_nodes * (20 + 4 * width) for j in group)
                assert metrics['grouped_prediction_unpack_bytes'] == prediction_plan(group).detached_bytes
                assert metrics['device_copy_bytes'] == metrics['grouped_prediction_pack_bytes'] + metrics['grouped_prediction_unpack_bytes']
                assert tuple(o.run_id for o in outcomes) == tuple(j.run_id for j in group)
                for j, o in zip(group, outcomes, strict=True):
                    assert ops._get(o.prediction, DevicePrediction).tree is j.tree and o.prediction.data is j.data
                observed = check(context, outcomes, expected, 'prediction')
                # Releasing earlier predictions leaves all later buffers intact.
                for outcome in outcomes:
                    np.testing.assert_array_equal(context.export(outcome.prediction.values), expected[outcome.run_id])
                    ops.release(outcome.prediction)
                assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
                schedules.append(dict(run_ids=[j.run_id for j in group], metrics=metrics, observed=observed))
            record = dict(
                count=count, width=width, codes=binned.codes.tolist(), missing=binned.missing.tolist(),
                input=snapshot(binned.data),
                jobs=[dict(run_id=j.run_id, topology=j.tree.topology, values=device_tree.export(ops, j.tree).value.tolist(),
                           model=device_tree.export(ops, j.tree).record()) for j in jobs],
                schedules=schedules, scope='grouped tree prediction only')
            path = save(tmp_path, f'prediction-M{count}-L{width}.json', record)
            fresh = subprocess.run(fresh_command(FRESH, str(path)), check=True, capture_output=True, text=True)
            record['fresh_inference'] = json.loads(fresh.stdout)
            assert record['fresh_inference'] == dict(models=count, predictions_matched=True, training_imports_denied=True)
            save(tmp_path, path.name, record)
        assert context.metrics['live_bytes'] == 0 and not ops._records


@pytest.mark.parametrize('operation', ['partition', 'prediction'])
@pytest.mark.parametrize('empty', [False, True])
def test_inactive_validation_and_empty_rows(operation, empty):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            routing, prediction, _, binned = setup(ops, empty=empty, make_trees=operation == 'prediction')
            jobs = routing if operation == 'partition' else prediction
            call = partitions if operation == 'partition' else predictions
            expected = expected_routes(context, jobs, binned) if operation == 'partition' else expected_predictions(ops, jobs, binned)
            before = dict(context.metrics)
            inactive = call(ops, jobs, active=(False,) * 3)
            assert [o.status for o in inactive] == ['inactive'] * 3 and dict(context.metrics) == before
            outcomes = call(ops, jobs, active=(True, False, True))
            assert outcomes[1].status == 'inactive'
            check(context, (outcomes[0], outcomes[2]), expected, operation)
            release(ops, outcomes, operation)


@pytest.mark.parametrize('fault', ['forged_rows', 'released_rows', 'released_split', 'released_candidates',
                                  'released_histogram', 'released_fields', 'released_data', 'foreign_ops'])
def test_partition_live_ancestry_even_for_inactive_slot(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _, _ = setup(ops, make_trees=False)
            job = jobs[1]
            if fault == 'forged_rows':
                jobs = (jobs[0], replace(job, rows=replace(job.rows)), jobs[2])
            elif fault.startswith('released_'):
                record = {'rows': job.rows, 'split': job.split, 'candidates': job.split.candidates,
                          'histogram': job.split.candidates.histogram, 'fields': job.split.candidates.histogram.fields,
                          'data': job.rows.data}[fault.removeprefix('released_')]
                ops.release(record)
            before = dict(context.metrics)
            with pytest.raises(ValueError):
                partitions(DeviceOperations(context) if fault == 'foreign_ops' else ops, jobs, active=(True, False, True))
            assert context.metrics['live_bytes'] == before['live_bytes'] and context.metrics['kernel_launches'] == before['kernel_launches']


@pytest.mark.parametrize('fault', ['forged_tree', 'released_tree', 'released_data', 'foreign_ops', 'separate_features', 'different_cuts'])
def test_prediction_live_inputs_even_for_inactive_slot(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            _, jobs, _, binned = setup(ops)
            job = jobs[1]
            if fault == 'forged_tree':
                jobs = (jobs[0], replace(job, tree=replace(job.tree)), jobs[2])
            elif fault in ('released_tree', 'released_data'):
                ops.release(getattr(job, fault.removeprefix('released_')))
            elif fault in ('separate_features', 'different_cuts'):
                other_binned = binned if fault == 'separate_features' else Binning.fit(binned.data, bins=2).transform(binned.data)
                p = Problem(binned.data, np.arange(7)[:, None], binned.data.row_ids)
                data = device_inputs.bind(ops, device_inputs.prepare(ops, other_binned), p)
                jobs = (jobs[0], replace(job, data=data), jobs[2])
            before = dict(context.metrics)
            with pytest.raises(ValueError):
                predictions(DeviceOperations(context) if fault == 'foreign_ops' else ops, jobs, active=(True, False, True))
            assert context.metrics['live_bytes'] == before['live_bytes'] and context.metrics['kernel_launches'] == before['kernel_launches']


@pytest.mark.parametrize('operation', ['partition', 'prediction'])
@pytest.mark.parametrize('stage', [1, 3, 5, 7])
def test_partial_allocation_failure_is_atomic_then_same_ids_retry(operation, stage, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            routing, prediction, _, binned = setup(ops, make_trees=operation == 'prediction')
            jobs, call = (routing, partitions) if operation == 'partition' else (prediction, predictions)
            expected = expected_routes(context, jobs, binned) if operation == 'partition' else expected_predictions(ops, jobs, binned)
            baseline, records, handles = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
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
                    call(ops, jobs)
            assert calls == stage and context.metrics['live_bytes'] == baseline
            assert set(ops._records) == records and set(context._buffers) == handles
            outcomes = call(ops, jobs)
            check(context, outcomes, expected, operation)
            release(ops, outcomes, operation)
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records


@pytest.mark.parametrize('operation', ['partition', 'prediction'])
def test_actual_pool_cap_failure_discards_scratch_and_retry_succeeds(operation):
    cap = 1024**2
    with ExecutionContext(max_bytes=cap) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            routing, prediction, _, binned = setup(ops, make_trees=operation == 'prediction')
            jobs, call = (routing, partitions) if operation == 'partition' else (prediction, predictions)
            expected = expected_routes(context, jobs, binned) if operation == 'partition' else expected_predictions(ops, jobs, binned)
            # Tree construction can leave reusable cached blocks. Release those
            # before filling the actual pool; peak bytes are historical, not the
            # currently occupied physical space after cache cleanup.
            context.synchronize()
            with context._scope():
                context._pool.free_all_blocks()
                occupied = context._pool.total_bytes()
                assert occupied == context._pool.used_bytes()
            scratch = context.upload(np.zeros((cap - occupied) // 4, np.float32))
            assert context.metrics['peak_pool_bytes'] == cap
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            with pytest.raises(MemoryError):
                call(ops, jobs)
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
            context.release(scratch)
            outcomes = call(ops, jobs)
            check(context, outcomes, expected, operation)
            release(ops, outcomes, operation)
            assert context.metrics['peak_pool_bytes'] <= cap


@pytest.mark.parametrize('count', [1, 8, 32])
def test_injected_nonfinite_prediction_isolates_slot_then_same_id_retry(count, monkeypatch, tmp_path):
    from openboost import device_group_tree as groups

    with ExecutionContext(max_bytes=8 * 1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            _, jobs, _, binned = setup(ops, count)
            expected = expected_predictions(ops, jobs, binned)
            bad = count // 2
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            launch = groups._launch

            def inject(ops, name, size, *args):
                launch(ops, name, size, *args)
                if name == 'predictions':
                    # Deliberate private fault injection tests status handling;
                    # finite validated leaf copies cannot naturally overflow.
                    args[-1][bad, 0, 0] = np.float32(np.nan)

            with monkeypatch.context() as patch:
                patch.setattr(groups, '_launch', inject)
                outcomes = predictions(ops, jobs)
            assert outcomes[bad].status == 'failed' and outcomes[bad].prediction is None
            assert outcomes[bad].error_type == 'ValueError'
            neighbors = check(context, tuple(o for o in outcomes if o.status == 'complete'), expected, 'prediction')
            assert len(neighbors) == count - 1
            release(ops, outcomes, 'prediction')
            retry = predictions(ops, (jobs[bad],))
            retried = check(context, retry, expected, 'prediction')
            release(ops, retry, 'prediction')
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
            save(tmp_path, f'prediction-injected-M{count}.json', dict(
                count=count, bad_run_id=jobs[bad].run_id,
                scope='injected nonfinite output; not finite-input overflow',
                codes=binned.codes.tolist(), missing=binned.missing.tolist(),
                jobs=[dict(run_id=j.run_id, topology=j.tree.topology, values=device_tree.export(ops, j.tree).value.tolist()) for j in jobs],
                statuses=[dict(run_id=o.run_id, status=o.status, error_type=o.error_type) for o in outcomes],
                neighbors=neighbors, retry=retried, input_live_bytes=baseline, after_release_live_bytes=context.metrics['live_bytes']))


def test_prediction_provenance_and_tree_release_preserve_independent_snapshot():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            _, jobs, _, binned = setup(ops)
            expected = expected_predictions(ops, jobs, binned)
            outcomes = predictions(ops, jobs)
            first = outcomes[0].prediction
            for forged in (replace(first), replace(first, tree=jobs[1].tree), replace(first, data=jobs[1].data)):
                with pytest.raises(ValueError, match='foreign|forged'):
                    ops._get(forged, DevicePrediction)
            ops.release(jobs[0].tree)
            assert ops._get(first, DevicePrediction) is first
            check(context, outcomes, expected, 'prediction')
            ops.release(first)
            with pytest.raises(ValueError, match='released'):
                context.export(first.values)
            check(context, outcomes[1:], expected, 'prediction')
            release(ops, outcomes[1:], 'prediction')


def test_grouped_histogram_partition_and_child_histogram_composition():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            routing, _, fields, binned = setup(ops, make_trees=False)
            jobs = tuple(HistogramJob(j.run_id, j.rows.data, field, j.rows)
                         for j, field in zip(routing, fields, strict=True))
            aggregates = histograms(ops, jobs)
            requests = []
            for j, outcome in zip(jobs, aggregates, strict=True):
                batch = ops.candidates(outcome.histogram)
                scores = np.zeros(batch.size, np.float32)
                scores[0] = 1
                split = ops.choose(batch, ops.scores(batch, context.upload(scores)),
                                   ops.mask(batch, context.upload(np.ones(batch.size, bool))))
                requests.append(PartitionJob(j.run_id, j.rows, split))
            children = partitions(ops, tuple(requests))
            expected = expected_routes(context, tuple(requests), binned)
            check(context, children, expected, 'partition')
            for side in (0, 1):
                grouped = tuple(HistogramJob(j.run_id, j.data, j.fields, child.children[side])
                                for j, child in zip(jobs, children, strict=True))
                outputs = histograms(ops, grouped)
                for job, output in zip(grouped, outputs, strict=True):
                    independent = ops.histogram(job.data, job.fields, job.rows)
                    for name in ('sums', 'counts', 'total'):
                        np.testing.assert_array_equal(context.export(getattr(output.histogram, name)),
                                                      context.export(getattr(independent, name)))
                    ops.release(independent)
                    ops.release(output.histogram)
