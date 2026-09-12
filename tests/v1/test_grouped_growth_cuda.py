"""138 real grouped growth, independent Newton controls and ownership failures."""

import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from openboost import device_group_growth as growth
from openboost import device_inputs, device_tree
from openboost.data import Problem
from openboost.device import DeviceOperations, _workspace
from openboost.execution import ExecutionContext
from openboost.stats import vector_newton

from .grouped_growth_artifacts import FRESH
from .grouped_tree_artifacts import snapshot
from .multi_squared_artifacts import fresh_command
from .reference import device_vector as ref
from .test_device_vector_ops_cuda import bind
from .test_device_vector_reference import prepared

pytestmark = pytest.mark.gpu


def setup(ops, count=3, width=2, projected=False):
    base, _, binned, _, _ = prepared(width, projected)
    shared = device_inputs.prepare(ops, binned)
    jobs, sources = [], {}
    for i in range(count):
        source = {k: v.copy() for k, v in base.items()}
        for key in ('g', 'split_g'):
            source[key] *= (i % 3 + 1) / 2
        source['weight'] *= i % 2 + 1
        problem = Problem(binned.data, np.zeros((8, 1)), binned.data.row_ids, weight=source['weight'])
        data = device_inputs.bind(ops, shared, problem)
        fields = bind(ops, data, ops.execution,
                      vector_newton(problem, source['split_g'], source['split_h']), reordered=True)
        leaves = bind(ops, data, ops.execution, vector_newton(problem, source['g'], source['h'])) if projected else None
        job = growth.TreeJob(f'run-{i}', data, fields, binned.binning, max_depth=(i + 2) % 3,
                             leaf_fields=leaves, output_width=width,
                             scoring=lambda o, b: o.vector_scores(b, reg_lambda=2, split_penalty=0.5),
                             legality=lambda o, b: o.vector_feasible(b, min_child_h=1),
                             leaf=lambda o, h: o.vector_leaf(h, reg_lambda=2))
        jobs.append(job)
        sources[job.run_id] = source
    return tuple(jobs), sources, binned


def sequential(ops, job):
    return device_tree.depthwise(ops, job.data, job.fields, binning=job.binning,
                                  max_depth=job.max_depth, leaf_fields=job.leaf_fields,
                                  output_width=job.output_width, scoring=job.scoring,
                                  legality=job.legality, leaf=job.leaf,
                                  reg_lambda=job.reg_lambda, min_child_h=job.min_child_h,
                                  split_penalty=job.split_penalty)


def record(ops, job, tree):
    model = device_tree.export(ops, tree)
    value = device_tree.predict(ops, tree, job.data)
    prediction = ops.execution.export(value).tolist()
    ops.execution.release(value)
    return dict(run_id=job.run_id, model=model.record(), prediction=prediction)


def controls(ops, jobs, sources):
    result = {}
    for job in jobs:
        tree = sequential(ops, job)
        known = ref.tree(sources[job.run_id], job.max_depth, regularization=2, penalty=0.5, minimum=1)
        model = device_tree.export(ops, tree)
        assert list(tree.topology) == [(*n['key'], n['left'], n['right']) if n['key'] else
                                      (-1, -1, False, -1, -1) for n in known]
        np.testing.assert_allclose(model.value, [[float(v) for v in n['value']] for n in known], rtol=1e-4, atol=1e-5)
        result[job.run_id] = record(ops, job, tree)
        ops.release(tree)
    return result


def forbidden(*args, **kwargs):
    raise AssertionError('grouped growth must use actual grouped histogram/routing operations')


def guard(patch, ops):
    patch.setattr(ops, 'histogram', forbidden)
    patch.setattr(ops, 'partition', forbidden)
    patch.setattr(device_tree, 'depthwise', forbidden)


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('width', [1, 2])
@pytest.mark.parametrize('projected', [False, True])
def test_grouped_trees_match_independent_runs_and_original_row_math(count, width, projected, monkeypatch, tmp_path):
    with ExecutionContext(max_bytes=16 * 1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, sources, binned = setup(ops, count, width, projected)
            expected = controls(ops, jobs, sources)
            baseline, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            schedules = []
            groups = ((jobs,), (jobs[::-1],), tuple(g for g in (jobs[::2], jobs[1::2]) if g))
            for schedule in groups:
                observed, metrics = [], []
                for group in schedule:
                    before = dict(context.metrics)
                    with monkeypatch.context() as patch:
                        guard(patch, ops)
                        outcomes = growth.depthwise(ops, group)
                    delta = {k: v - before.get(k, 0) for k, v in context.metrics.items() if isinstance(v, (int, float))}
                    assert [o.run_id for o in outcomes] == [j.run_id for j in group]
                    assert all(o.status == 'complete' for o in outcomes)
                    node_count = sum(o.tree.n_nodes for o in outcomes)
                    assert delta['grouped_growth_node_slots'] == node_count
                    assert delta['grouped_histogram_slots'] >= node_count
                    assert delta['grouped_growth_phases'] == max(o.tree.n_nodes for o in outcomes)
                    internal = sum(sum(n[0] != -1 for n in o.tree.topology) for o in outcomes)
                    assert delta.get('grouped_partition_slots', 0) == internal
                    assert set(ops._records) - records == {o.tree for o in outcomes}
                    assert set(context._buffers) - buffers == {h for o in outcomes for h in ops._records[o.tree][0]}
                    for job, outcome in zip(group, outcomes, strict=True):
                        actual = record(ops, job, outcome.tree)
                        assert actual == expected[job.run_id]
                        observed.append(actual)
                        ops.release(outcome.tree)
                    metrics.append(delta)
                    assert context.metrics['live_bytes'] == baseline
                    assert set(ops._records) == records and set(context._buffers) == buffers
                schedules.append(dict(groups=[[j.run_id for j in g] for g in schedule], observed=observed, metrics=metrics))
            payload = dict(count=count, width=width, projected=projected, input=snapshot(binned.data),
                           jobs=[dict(run_id=j.run_id, depth=j.max_depth) for j in jobs], schedules=schedules,
                           input_live_bytes=baseline, final_live_bytes=context.metrics['live_bytes'])
            root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'grouped-growth'
            root.mkdir(parents=True, exist_ok=True)
            path = root / f'M{count}-L{width}-P{int(projected)}.json'
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
            fresh = subprocess.run(fresh_command(FRESH, str(path)), capture_output=True, text=True, check=True, timeout=60)
            payload['fresh'] = json.loads(fresh.stdout)
            assert payload['fresh']['models'] == 3 * count
            path.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


@pytest.mark.parametrize('fault', ['leaf', 'scoring', 'legality', 'late_leaf', 'nonfinite_leaf', 'histogram_overflow'])
def test_one_failed_run_releases_its_work_and_same_id_retries(fault, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, sources, _ = setup(ops)
            jobs = tuple(replace(j, max_depth=2) for j in jobs)
            expected = controls(ops, jobs, sources)
            bad, calls = jobs[1], 0

            def fail(o, h):
                nonlocal calls
                calls += 1
                # A newly allocated callback scratch buffer must not leak.
                context.upload(np.ones(17, np.float32))
                if fault == 'late_leaf' and calls < 2:
                    return o.vector_leaf(h, reg_lambda=2)
                if fault == 'nonfinite_leaf':
                    return context.upload(np.array([np.nan, 0], np.float32))
                raise ValueError('declared per-run domain failure')

            if fault == 'histogram_overflow':
                values = context.upload(np.full((8, len(bad.fields.names)), 3e38, np.float32))
                bad = replace(bad, fields=ops.fields(bad.data, values, names=bad.fields.names, roles=bad.fields.roles))
            else:
                key = fault if fault in ('scoring', 'legality') else 'leaf'
                bad = replace(bad, **{key: fail})
            submitted = (jobs[0], bad, jobs[2])
            baseline, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            with monkeypatch.context() as patch:
                guard(patch, ops)
                outcomes = growth.depthwise(ops, submitted)
            assert [o.status for o in outcomes] == ['complete', 'failed', 'complete']
            assert outcomes[1].error_type == 'ValueError' and outcomes[1].tree is None
            if fault == 'late_leaf':
                assert calls == 2
            for i in (0, 2):
                assert record(ops, jobs[i], outcomes[i].tree) == expected[jobs[i].run_id]
                ops.release(outcomes[i].tree)
            assert context.metrics['live_bytes'] == baseline
            assert set(ops._records) == records and set(context._buffers) == buffers
            retry = growth.depthwise(ops, (jobs[1],))[0]
            assert retry.status == 'complete'
            assert record(ops, jobs[1], retry.tree) == expected[jobs[1].run_id]
            ops.release(retry.tree)
            assert context.metrics['live_bytes'] == baseline


@pytest.mark.parametrize('stage', [1, 3, 7, 15, 35])
def test_partial_allocation_failure_is_atomic_and_preserves_retry(stage, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, sources, _ = setup(ops)
            expected = controls(ops, jobs, sources)
            baseline, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            original, calls = context._empty, 0

            def fail(shape, dtype):
                nonlocal calls
                calls += 1
                if calls == stage:
                    raise MemoryError('declared grouped growth allocation failure')
                return original(shape, dtype)

            with monkeypatch.context() as patch:
                patch.setattr(context, '_empty', fail)
                with pytest.raises(MemoryError, match='declared grouped growth'):
                    growth.depthwise(ops, jobs)
            assert calls == stage and context.metrics['live_bytes'] == baseline
            assert set(ops._records) == records and set(context._buffers) == buffers
            for job, outcome in zip(jobs, growth.depthwise(ops, jobs), strict=True):
                assert record(ops, job, outcome.tree) == expected[job.run_id]
                ops.release(outcome.tree)
            assert context.metrics['live_bytes'] == baseline


@pytest.mark.parametrize('fault', ['forged_data', 'forged_fields', 'released_data', 'released_fields', 'released_values', 'foreign_ops'])
def test_all_live_inputs_checked_even_for_inactive_run(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _ = setup(ops)
            bad = jobs[1]
            if fault == 'forged_data':
                data = replace(bad.data)
                bad = replace(bad, data=data, fields=replace(bad.fields, data=data))
            elif fault == 'forged_fields':
                bad = replace(bad, fields=replace(bad.fields))
            elif fault == 'released_values':
                context.release(bad.fields.values)
            elif fault.startswith('released_'):
                ops.release(getattr(bad, fault.removeprefix('released_')))
            submitted = (jobs[0], bad, jobs[2])
            before = dict(context.metrics)
            with pytest.raises(ValueError):
                growth.depthwise(DeviceOperations(context) if fault == 'foreign_ops' else ops,
                                 submitted, active=(True, False, True))
            assert context.metrics['live_bytes'] == before['live_bytes']
            assert context.metrics['kernel_launches'] == before['kernel_launches']


def test_inactive_jobs_allocate_nothing_and_mixed_mask_keeps_caller_order():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, sources, _ = setup(ops)
            expected = controls(ops, jobs, sources)
            before = dict(context.metrics)
            inactive = growth.depthwise(ops, jobs, active=(False,) * 3)
            assert [o.status for o in inactive] == ['inactive'] * 3
            assert dict(context.metrics) == before
            mixed = growth.depthwise(ops, jobs, active=(True, False, True))
            assert [o.status for o in mixed] == ['complete', 'inactive', 'complete']
            for i in (0, 2):
                assert record(ops, jobs[i], mixed[i].tree) == expected[jobs[i].run_id]
                ops.release(mixed[i].tree)
            assert context.metrics['live_bytes'] == before['live_bytes']


def test_actual_pool_cap_rejects_scratch_then_same_group_retries():
    cap = 2 * 1024**2
    with ExecutionContext(max_bytes=cap) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, sources, _ = setup(ops)
            expected = controls(ops, jobs, sources)
            context.synchronize()
            with context._scope():
                context._pool.free_all_blocks()
                occupied = context._pool.total_bytes()
                assert occupied == context._pool.used_bytes()
            scratch = context.upload(np.zeros((cap - occupied) // 4, np.float32))
            baseline, records = context.metrics['live_bytes'], set(ops._records)
            with pytest.raises(MemoryError):
                growth.depthwise(ops, jobs)
            assert context.metrics['live_bytes'] == baseline and set(ops._records) == records
            context.release(scratch)
            for job, outcome in zip(jobs, growth.depthwise(ops, jobs), strict=True):
                assert record(ops, job, outcome.tree) == expected[job.run_id]
                ops.release(outcome.tree)
            assert context.metrics['peak_pool_bytes'] <= cap


@pytest.mark.parametrize('width', [1, 2, 4])
def test_public_assembly_owns_values_after_input_release(width):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            _, _, binned = setup(ops, 1)
            topology = ((0, 0, True, 1, 2), (-1, -1, False, -1, -1), (-1, -1, False, -1, -1))
            values = tuple(context.upload(np.arange(width, dtype=np.float32) + i) for i in range(3))
            baseline = context.metrics['live_bytes']
            tree = device_tree.assemble(ops, binning=binned.binning, topology=topology, values=values, output_width=width)
            assert context.metrics['live_bytes'] - baseline == 3 * (20 + 4 * width)
            expected = device_tree.export(ops, tree).record()
            for value in values:
                context.release(value)
            assert device_tree.export(ops, tree).record() == expected
            ops.release(tree)


@pytest.mark.parametrize('fault', ['empty', 'cycle', 'wrong_count', 'wrong_shape', 'released', 'nonfinite'])
def test_invalid_public_assembly_is_atomic(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            _, _, binned = setup(ops, 1)
            topology = ((-1, -1, False, -1, -1),)
            values = (context.upload(np.array([1, 2], np.float32)),)
            if fault == 'empty':
                topology = ()
            elif fault == 'cycle':
                topology = ((0, 0, True, 0, 0),)
            elif fault == 'wrong_count':
                values = values * 2
            elif fault == 'wrong_shape':
                values = (context.upload(np.ones(1, np.float32)),)
            elif fault == 'released':
                context.release(values[0])
            else:
                values = (context.upload(np.array([np.inf, 0], np.float32)),)
            baseline, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            with pytest.raises(ValueError):
                device_tree.assemble(ops, binning=binned.binning, topology=topology, values=values, output_width=2)
            assert context.metrics['live_bytes'] == baseline
            assert set(ops._records) == records and set(context._buffers) == buffers


def test_common_histogram_layout_allows_distinct_output_widths():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _ = setup(ops)
            def constant(width):
                return lambda o, h: o.execution.upload(np.arange(width, dtype=np.float32))
            jobs = tuple(replace(j, output_width=i + 1, leaf=constant(i + 1)) for i, j in enumerate(jobs))
            expected = {}
            for job in jobs:
                tree = sequential(ops, job)
                expected[job.run_id] = record(ops, job, tree)
                ops.release(tree)
            for job, outcome in zip(jobs, growth.depthwise(ops, jobs), strict=True):
                assert outcome.status == 'complete'
                assert record(ops, job, outcome.tree) == expected[job.run_id]
                ops.release(outcome.tree)


@pytest.mark.parametrize('depth', [0, 2])
def test_default_scalar_policy_composes_without_callbacks(depth):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _ = setup(ops, width=1)
            scalar = []
            for job in jobs:
                # setup's split columns are reversed: curvature then gradient.
                fields = ops.fields(job.data, job.fields.values, names=('curvature', 'gradient'),
                                    roles=('training', 'training'))
                scalar.append(replace(job, fields=fields, max_depth=depth, scoring=None,
                                      legality=None, leaf=None, reg_lambda=2, split_penalty=0.5))
            jobs = tuple(scalar)
            expected = {}
            for job in jobs:
                tree = sequential(ops, job)
                expected[job.run_id] = record(ops, job, tree)
                ops.release(tree)
            for job, outcome in zip(jobs, growth.depthwise(ops, jobs), strict=True):
                assert outcome.status == 'complete'
                assert record(ops, job, outcome.tree) == expected[job.run_id]
                ops.release(outcome.tree)


def test_unexpected_failure_discards_already_completed_neighbor(monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            jobs, _, _ = setup(ops)
            original = jobs[0].leaf
            calls = 0

            def fail_later(o, h):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError('unexpected callback failure')
                return original(o, h)

            jobs = (replace(jobs[0], leaf=fail_later), replace(jobs[1], max_depth=0), jobs[2])
            baseline, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            with pytest.raises(RuntimeError, match='unexpected callback'):
                growth.depthwise(ops, jobs)
            assert calls == 2
            assert context.metrics.get('grouped_growth_successes', 0) >= 1
            assert context.metrics['live_bytes'] == baseline
            assert set(ops._records) == records and set(context._buffers) == buffers
