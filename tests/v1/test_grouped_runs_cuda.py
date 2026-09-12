"""140 complete grouped squared schedules, independent outcomes and fault ownership."""

import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from openboost import (
    device_group_growth,
    device_group_runs,
    device_group_tree,
    device_runs,
    device_tree,
)
from openboost.device import DeviceOperations, _workspace
from openboost.device_active import SquaredPhase
from openboost.device_runtime import DeviceRun
from openboost.execution import ExecutionContext
from openboost.runtime import RunContext

from .glm_artifacts import input_snapshot
from .test_device_active_cuda import forbidden, prepare
from .test_device_runs_cuda import check_reference, fresh_inference, record
from .test_scheduler_policy_reference import mixed

pytestmark = pytest.mark.gpu


def baseline(ops, specs):
    outcomes = device_runs.run_many(ops, specs)
    assert all(o.result is not None for o in outcomes)
    return {o.run_id: o.result for o in outcomes}


def rows(specs, results):
    return [dict(run_id=s.run_id, seed=s.seed, options=dict(s.options), result=record(results[s.run_id]),
                 inputs=dict(train=input_snapshot(s.train), validation=input_snapshot(s.validation)),
                 predictions={key: {split: getattr(results[s.run_id], key).predict(p.data, offset=p.offset).tolist()
                                    for split, p in (('train', s.train), ('validation', s.validation))}
                              for key in ('model', 'best_model')}) for s in specs]


def observed(outcomes):
    return [dict(run_id=o.run_id, result=None if o.result is None else record(o.result),
                 error_type=o.error_type, error_message=o.error_message) for o in outcomes]


def retained(payload, name, tmp_path):
    root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'grouped-squared'
    root.mkdir(parents=True, exist_ok=True)
    fresh_inference(payload, root / (name+'.json'))


def unchanged(ops, before):
    live, records, buffers = before
    assert ops.execution.metrics['live_bytes'] == live
    assert set(ops._records) == records and set(ops.execution._buffers) == buffers


def snapshot(ops):
    return ops.execution.metrics['live_bytes'], set(ops._records), set(ops.execution._buffers)


@pytest.mark.parametrize('count', [1, 8, 32])
def test_complete_scheduler_matches_independent_recipe_permutations_regrouping_and_fresh_inference(count, monkeypatch, tmp_path):
    with ExecutionContext(max_bytes=16*1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, count)
            expected = baseline(ops, specs)
            for s in specs:
                check_reference(s, expected[s.run_id])
            before, schedules = snapshot(ops), []
            for submitted, size in ((specs, 32), (specs[::-1], 32), (specs[::2]+specs[1::2], 3)):
                groups, rounds = [], []
                grow, advance = device_group_growth.depthwise, SquaredPhase.advance
                metrics = dict(context.metrics)

                def grow_observed(o, jobs, *, _groups=groups, _grow=grow, **kw):
                    _groups.append([j.run_id for j in jobs])
                    return _grow(o, jobs, **kw)

                def advance_observed(phase, *args, _rounds=rounds, _advance=advance):
                    index, name = phase.stop.completed_rounds, phase.run.run_id
                    rng = phase.run.rng(index, 'learner', 'rows').integers(0, 2**32, 8).tolist()
                    assert rng == RunContext(name, phase.run.seed).rng(index, 'learner', 'rows').integers(0, 2**32, 8).tolist()
                    step = _advance(phase, *args)
                    _rounds.append(dict(run_id=name, round=index, step=asdict(step), stop=asdict(phase.stop), rng=rng))
                    return step

                with monkeypatch.context() as patch:
                    patch.setattr(device_group_growth, 'depthwise', grow_observed)
                    patch.setattr(SquaredPhase, 'advance', advance_observed)
                    for target, name in ((device_runs, 'run_many'), (device_tree, 'depthwise'), (device_tree, 'predict'), (ops, 'histogram'), (ops, 'partition')):
                        patch.setattr(target, name, forbidden)
                    actual = device_group_runs.run_many(ops, submitted, group_size=size)
                assert [o.run_id for o in actual] == [s.run_id for s in submitted]
                assert all(o.error_type is o.error_message is None for o in actual)
                assert {o.run_id: record(o.result) for o in actual} == {n: record(r) for n, r in expected.items()}
                unchanged(ops, before)
                delta = {k: v-metrics.get(k, 0) for k, v in context.metrics.items() if isinstance(v, (int, float))}
                assert delta['grouped_growth_calls'] == delta['grouped_run_groups'] == len(groups)
                assert delta['grouped_prediction_calls'] == 2*len(groups)
                assert delta['grouped_run_completions'] == count and delta.get('grouped_run_failures', 0) == 0
                plan = device_group_runs.schedule_plan(submitted, group_size=size)
                schedules.append(dict(order=[list(g) for g in plan.groups], groups=groups, rounds=rounds,
                                      results={o.run_id: record(o.result) for o in actual}, metrics=delta, group_size=size))
            retained(dict(count=count, schedules=schedules, outcomes=rows(specs, expected),
                          input_live_bytes=before[0], final_live_bytes=context.metrics['live_bytes']), f'M{count}', tmp_path)


def phase_failure(count, stage, monkeypatch, tmp_path):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, count)
            expected = baseline(ops, specs)
            bad, before = specs[count//2].run_id, snapshot(ops)
            if stage == 'initialize':
                original = DeviceRun.initialize

                def fail(run):
                    result = original(run)
                    if run.run_id == bad:
                        raise MemoryError('declared scheduler initialization failure')
                    return result

                target, method = DeviceRun, 'initialize'
            else:
                target, method = SquaredPhase, {'request': 'request_tree', 'advance': 'advance', 'export': 'result'}[stage]
                original = getattr(target, method)

                def fail(phase, *args):
                    if phase.run.run_id == bad and (stage != 'advance' or phase.stop.completed_rounds == 1):
                        with _workspace(ops):
                            context.upload(np.ones(7, np.float32))
                            raise ValueError('declared scheduler '+stage+' failure')
                    return original(phase, *args)
            with monkeypatch.context() as patch:
                patch.setattr(target, method, fail)
                actual = device_group_runs.run_many(ops, specs)
            unchanged(ops, before)
            for o in actual:
                if o.run_id == bad:
                    assert o.result is None and o.error_type == ('MemoryError' if stage == 'initialize' else 'ValueError')
                    assert o.error_message == 'declared scheduler '+('initialization' if stage == 'initialize' else stage)+' failure'
                else:
                    assert record(o.result) == record(expected[o.run_id]) and o.error_type is None
            retry = device_group_runs.run_many(ops, (specs[count//2],))
            assert record(retry[0].result) == record(expected[bad])
            unchanged(ops, before)
            retained(dict(count=count, stage=stage, failed_ids=[bad], observed=observed(actual), retry=observed(retry),
                          outcomes=rows(specs, expected), input_live_bytes=before[0], final_live_bytes=context.metrics['live_bytes']),
                     f'failure-M{count}-{stage}', tmp_path)


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('stage', ['initialize', 'advance'])
def test_one_failed_run_preserves_neighbors_and_same_id_retry(count, stage, monkeypatch, tmp_path):
    phase_failure(count, stage, monkeypatch, tmp_path)


@pytest.mark.parametrize('stage', ['request', 'export'])
def test_request_and_export_failure_preserve_neighbors(stage, monkeypatch, tmp_path):
    phase_failure(3, stage, monkeypatch, tmp_path)


@pytest.mark.parametrize('size', [2, 32])
@pytest.mark.parametrize('stage', ['growth', 'train_prediction', 'validation_prediction'])
def test_shared_operation_failure_affects_only_submitted_active_group(size, stage, monkeypatch, tmp_path):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 5)
            specs = specs[:-1]+(replace(specs[-1], options=dict(specs[-1].options, rounds=0)),)
            expected, before = baseline(ops, specs), snapshot(ops)
            target = device_group_growth if stage == 'growth' else device_group_tree
            method = 'depthwise' if stage == 'growth' else 'predictions'
            original, calls, failed = getattr(target, method), 0, []

            def fail(o, jobs, **kwargs):
                nonlocal calls
                calls += 1
                if calls == (2 if stage == 'validation_prediction' else 1):
                    failed.extend(j.run_id for j in jobs)
                    with _workspace(o):
                        context.upload(np.ones(7, np.float32))
                        raise RuntimeError('declared shared '+stage+' failure')
                return original(o, jobs, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(target, method, fail)
                actual = device_group_runs.run_many(ops, specs, group_size=size)
            assert failed == [s.run_id for s in specs[:min(size, 4)]]
            unchanged(ops, before)
            for outcome in actual:
                if outcome.run_id in failed:
                    assert outcome.result is None and outcome.error_type == 'RuntimeError'
                else:
                    assert record(outcome.result) == record(expected[outcome.run_id])
            retry = device_group_runs.run_many(ops, tuple(s for s in specs if s.run_id in failed), group_size=size)
            assert all(record(o.result) == record(expected[o.run_id]) for o in retry)
            unchanged(ops, before)
            retained(dict(count=5, stage=stage, group_size=size, failed_ids=failed, observed=observed(actual), retry=observed(retry),
                          outcomes=rows(specs, expected), input_live_bytes=before[0], final_live_bytes=context.metrics['live_bytes']),
                     f'shared-{size}-{stage}', tmp_path)


@pytest.mark.parametrize('index', [1, 3, 7, 15, 31])
def test_partial_scheduler_allocations_leave_no_owned_state_and_retry(index, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 3)
            expected, before = baseline(ops, specs), snapshot(ops)
            original, calls = context._empty, 0

            def fail(shape, dtype):
                nonlocal calls
                calls += 1
                if calls == index:
                    raise MemoryError('declared scheduler allocation failure')
                return original(shape, dtype)

            with monkeypatch.context() as patch:
                patch.setattr(context, '_empty', fail)
                actual = device_group_runs.run_many(ops, specs)
            assert calls >= index and any(o.error_type == 'MemoryError' for o in actual)
            assert all(o.result is None or record(o.result) == record(expected[o.run_id]) for o in actual)
            unchanged(ops, before)
            retry = device_group_runs.run_many(ops, specs)
            assert all(record(o.result) == record(expected[o.run_id]) for o in retry)
            unchanged(ops, before)


def test_real_physical_pool_cap_returns_failures_then_same_specs_rebind():
    cap = 2*1024**2
    with ExecutionContext(max_bytes=cap) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 3)
            expected = baseline(ops, specs)
            with context._scope():
                context._pool.free_all_blocks()
                occupied = context._pool.total_bytes()
                assert occupied == context._pool.used_bytes()
            scratch = context.upload(np.zeros((cap-occupied)//4, np.float32))
            before = snapshot(ops)
            actual = device_group_runs.run_many(ops, specs)
            assert all(o.result is None and o.error_type in ('MemoryError', 'OutOfMemoryError') for o in actual)
            unchanged(ops, before)
            context.release(scratch)
            retry = device_group_runs.run_many(ops, specs)
            assert all(record(o.result) == record(expected[o.run_id]) for o in retry)
            assert context.metrics['peak_pool_bytes'] <= cap


@pytest.mark.parametrize('fault', ['forged', 'record', 'codes', 'missing'])
def test_invalid_shared_feature_registration_rejects_before_owned_allocation(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, features = prepare(ops, 2)
            if fault == 'forged':
                features = (replace(features[0]), features[1])
                specs = tuple(replace(s, prepared=features) for s in specs)
            elif fault == 'record':
                ops.release(features[0])
            else:
                context.release(getattr(features[0], fault))
            before = snapshot(ops)
            with pytest.raises(ValueError):
                device_group_runs.run_many(ops, specs)
            unchanged(ops, before)


@pytest.mark.parametrize('stage', ['growth', 'advance'])
def test_interrupt_propagates_after_releasing_all_owned_runs(stage, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 3)
            before = snapshot(ops)

            def interrupt(*args, **kwargs):
                context.upload(np.ones(7, np.float32))
                raise KeyboardInterrupt('declared scheduler interruption')

            with monkeypatch.context() as patch:
                patch.setattr(device_group_growth if stage == 'growth' else SquaredPhase,
                              'depthwise' if stage == 'growth' else 'advance', interrupt)
                with pytest.raises(KeyboardInterrupt, match='declared scheduler'):
                    device_group_runs.run_many(ops, specs)
            unchanged(ops, before)


@pytest.mark.parametrize('count', [1, 8])
def test_zero_rounds_export_and_release_without_tree_or_prediction_work(count, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, count)
            specs = tuple(replace(s, options=dict(s.options, rounds=0)) for s in specs)
            expected, before = baseline(ops, specs), snapshot(ops)
            with monkeypatch.context() as patch:
                patch.setattr(device_group_growth, 'depthwise', forbidden)
                patch.setattr(device_group_tree, 'predictions', forbidden)
                actual = device_group_runs.run_many(ops, specs)
            assert all(record(o.result) == record(expected[o.run_id]) for o in actual)
            unchanged(ops, before)


def test_mixed_depths_trial_counts_and_policies_match_native_independent_fits(tmp_path):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 6)
            specs = mixed(specs)
            expected, before = baseline(ops, specs), snapshot(ops)
            schedules = []
            for size in (2, 32):
                actual = device_group_runs.run_many(ops, specs, group_size=size)
                assert all(record(o.result) == record(expected[o.run_id]) for o in actual)
                unchanged(ops, before)
                schedules.append(dict(group_size=size, outcomes=observed(actual)))
            retained(dict(count=6, schedules=schedules, outcomes=rows(specs, expected),
                          input_live_bytes=before[0], final_live_bytes=context.metrics['live_bytes']), 'mixed-policy', tmp_path)
