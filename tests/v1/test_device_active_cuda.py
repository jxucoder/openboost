"""139 active squared phases: real interleaving, trials and independent lifetime."""

import os
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from openboost import device_group_growth, device_inputs, device_recipes, device_tree
from openboost.device import DeviceFields, DeviceOperations, _workspace
from openboost.device_active import SquaredConfiguration, SquaredPhase
from openboost.device_group_tree import PredictionJob, predictions
from openboost.device_runs import run_many
from openboost.execution import ExecutionContext
from openboost.runtime import RunContext

from .glm_artifacts import input_snapshot
from .test_device_runs_cuda import check_reference, fresh_inference, record
from .test_device_runs_reference import jobs

pytestmark = pytest.mark.gpu


def prepare(ops, count=3):
    specs = jobs(count)
    first = specs[0]
    features = tuple(device_inputs.prepare(ops, first.binning.transform(p.data)) for p in (first.train, first.validation))
    return tuple(replace(s, prepared=features) for s in specs), features


def start(ops, spec, **options):
    return SquaredPhase(ops, spec.train, spec.validation, run_id=spec.run_id, seed=spec.seed,
                        binning=spec.binning, prepared=spec.prepared,
                        configuration=SquaredConfiguration(**(dict(spec.options) | options)))


def pair(ops, phase, tree):
    a = predictions(ops, (PredictionJob(phase.run.run_id, tree, phase.run.data),))[0]
    b = predictions(ops, (PredictionJob(phase.run.run_id, tree, phase.run.validation_data),))[0]
    assert a.status == b.status == 'complete'
    return a.prediction, b.prediction


def learner(ops, phase):
    request = phase.request_tree()
    try:
        outcome = device_group_growth.depthwise(ops, (request,))[0]
        assert outcome.status == 'complete'
        return outcome.tree
    finally:
        ops.release(request.fields)


def forbidden(*args, **kwargs):
    raise AssertionError('active grouped consumer must use grouped operations and registered predictions')


@pytest.mark.parametrize('count', [1, 8, 32])
def test_interleaved_phases_match_independent_recipe_and_original_row_reference(count, monkeypatch, tmp_path):
    with ExecutionContext(max_bytes=16 * 1024**2) as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, count)
            expected_outcomes = run_many(ops, specs)
            assert all(o.result is not None for o in expected_outcomes)
            expected = {o.run_id: o.result for o in expected_outcomes}
            for spec in specs:
                check_reference(spec, expected[spec.run_id])
            baseline, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            schedules, results, streams = [], {}, {}
            for schedule in ((specs,), (specs[::-1],), tuple(g for g in (specs[::2], specs[1::2]) if g)):
                before_metrics = dict(context.metrics)
                phases = {s.run_id: start(ops, s) for s in specs}
                observed_groups, rounds = [], []
                try:
                    with monkeypatch.context() as patch:
                        patch.setattr(device_tree, 'depthwise', forbidden)
                        patch.setattr(device_tree, 'predict', forbidden)
                        patch.setattr(ops, 'histogram', forbidden)
                        patch.setattr(ops, 'partition', forbidden)
                        while any(p.active for p in phases.values()):
                            for group in schedule:
                                active = [phases[s.run_id] for s in group if phases[s.run_id].active]
                                if not active:
                                    continue
                                ids = [p.run.run_id for p in active]
                                observed_groups.append(ids)
                                requests = tuple(p.request_tree() for p in active)
                                outcomes = device_group_growth.depthwise(ops, requests)
                                assert all(o.status == 'complete' for o in outcomes)
                                for request in requests:
                                    ops.release(request.fields)
                                a = predictions(ops, tuple(PredictionJob(p.run.run_id, o.tree, p.run.data)
                                                           for p, o in zip(active, outcomes, strict=True)))
                                b = predictions(ops, tuple(PredictionJob(p.run.run_id, o.tree, p.run.validation_data)
                                                           for p, o in zip(active, outcomes, strict=True)))
                                assert all(o.status == 'complete' for o in (*a, *b))
                                for phase, tree, train, validation in zip(active, outcomes, a, b, strict=True):
                                    index, run_id = phase.stop.completed_rounds, phase.run.run_id
                                    rng = phase.run.rng(index, 'learner', 'rows').integers(0, 2**32, 8).tolist()
                                    assert rng == RunContext(run_id, phase.run.seed).rng(index, 'learner', 'rows').integers(0, 2**32, 8).tolist()
                                    assert streams.setdefault((run_id, index), rng) == rng
                                    step = phase.advance(train.prediction, validation.prediction)
                                    assert phase.stop.completed_rounds == index + 1
                                    assert len(phase.steps) == index + 1 and phase.steps[-1] is step
                                    rounds.append(dict(run_id=run_id, round=index, step=asdict(step), stop=asdict(phase.stop), rng=rng))
                                    assert not phase.run._proposals
                                    for item in (train.prediction, validation.prediction, tree.tree):
                                        ops.release(item)
                    current = {name: p.result() for name, p in phases.items()}
                    assert {name: record(r) for name, r in current.items()} == {name: record(r) for name, r in expected.items()}
                    for name, result in current.items():
                        if name in results:
                            assert record(result) == record(results[name])
                        results[name] = result
                    schedules.append(dict(order=[[s.run_id for s in group] for group in schedule], groups=observed_groups,
                                          rounds=rounds, results={name: record(r) for name, r in current.items()}))
                finally:
                    for phase in phases.values():
                        phase.close()
                assert context.metrics['live_bytes'] == baseline
                assert set(ops._records) == records and set(context._buffers) == buffers
                metrics = {k: v - before_metrics.get(k, 0) for k, v in context.metrics.items() if isinstance(v, (int, float))}
                assert metrics['grouped_growth_calls'] == len(observed_groups)
                assert metrics['grouped_prediction_calls'] == 2 * len(observed_groups)
                schedules[-1]['metrics'] = metrics
            retained = []
            for spec in specs:
                result = results[spec.run_id]
                retained.append(dict(run_id=spec.run_id, seed=spec.seed, options=dict(spec.options),
                                     result=record(result), inputs=dict(train=input_snapshot(spec.train), validation=input_snapshot(spec.validation)),
                                     predictions={key: {split: getattr(result, key).predict(p.data, offset=p.offset).tolist()
                                                        for split, p in (('train', spec.train), ('validation', spec.validation))}
                                                  for key in ('model', 'best_model')}))
            payload = dict(count=count, schedules=schedules, outcomes=retained,
                           input_live_bytes=baseline, final_live_bytes=context.metrics['live_bytes'])
            root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'active-squared'
            root.mkdir(parents=True, exist_ok=True)
            path = root / f'M{count}.json'
            fresh_inference(payload, path)


@pytest.mark.parametrize('step', ['fixed', 'backtracking'])
def test_zero_rounds_and_terminal_guards_preserve_state(step):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 1)
            phase = start(ops, specs[0], rounds=0, step=step)
            assert not phase.active and phase.stop.reason == 'budget' and phase.steps == ()
            result = phase.result()
            assert result.state.n_terms == result.state.best_n_terms == 0
            before = dict(context.metrics)
            for action in (phase.request_tree, lambda: phase.advance(None, None)):
                with pytest.raises(ValueError, match='terminal'):
                    action()
            assert dict(context.metrics) == before
            phase.close()
            phase.close()
            with pytest.raises(RuntimeError, match='closed'):
                phase.result()


@pytest.mark.parametrize('fault', ['forged', 'released_prediction', 'released_values', 'released_tree',
                                  'foreign_data', 'swapped', 'raw_buffer', 'different_tree', 'closed'])
def test_foreign_or_dead_predictions_reject_before_trials(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 2)
            phases = [start(ops, s, rounds=1) for s in specs]
            tree = learner(ops, phases[0])
            a, b = pair(ops, phases[0], tree)
            if fault == 'forged':
                a = replace(a)
            elif fault == 'released_prediction':
                ops.release(a)
            elif fault == 'released_values':
                context.release(a.values)
            elif fault == 'released_tree':
                ops.release(tree)
            elif fault == 'foreign_data':
                a = predictions(ops, (PredictionJob('other', tree, phases[1].run.data),))[0].prediction
            elif fault == 'swapped':
                a, b = b, a
            elif fault == 'raw_buffer':
                a = a.values
            elif fault == 'different_tree':
                copied = device_tree.copy(ops, tree)
                b = pair(ops, phases[0], copied)[1]
            else:
                phases[0].close()
            phase = phases[0]
            old_state, old_stop, old_steps, serial = phase._state, phase._stop, tuple(phase._steps), phase._run._serial
            before = dict(context.metrics)
            with pytest.raises((ValueError, RuntimeError)):
                phase.advance(a, b)
            assert phase._state is old_state and phase._stop is old_stop and tuple(phase._steps) == old_steps
            assert phase._run._serial == serial
            assert context.metrics['live_bytes'] == before['live_bytes'] and context.metrics['kernel_launches'] == before['kernel_launches']
            # An independent live phase is still able to consume its own request.
            neighbor = learner(ops, phases[1])
            train, valid = pair(ops, phases[1], neighbor)
            phases[1].advance(train, valid)
            assert not phases[1].active
            phases[1].result()
            for p in phases:
                p.close()


@pytest.mark.parametrize('stage', [1, 3, 5, 7])
def test_failed_trial_allocations_preserve_parent_and_can_retry(stage, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 1)
            phase = start(ops, specs[0], rounds=1)
            tree = learner(ops, phase)
            train, valid = pair(ops, phase, tree)
            state, stop, serial = phase.state, phase.stop, phase.run._serial
            before, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            original, calls = context._empty, 0

            def fail(shape, dtype):
                nonlocal calls
                calls += 1
                if calls == stage:
                    raise MemoryError('declared active trial allocation failure')
                return original(shape, dtype)

            with monkeypatch.context() as patch:
                patch.setattr(context, '_empty', fail)
                with pytest.raises(MemoryError, match='declared active'):
                    phase.advance(train, valid)
            assert calls == stage
            assert phase.state is state and phase.stop is stop and phase.steps == () and phase.run._serial == serial
            assert context.metrics['live_bytes'] == before and set(ops._records) == records and set(context._buffers) == buffers
            phase.advance(train, valid)
            assert not phase.active
            phase.result()
            phase.close()


def test_repeated_tree_requests_and_unfinished_export_preserve_other_phases():
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 2)
            first, second = (start(ops, s, rounds=1) for s in specs)
            a, b = first.request_tree(), first.request_tree()
            assert a.fields is not b.fields and a.fields.values is not b.fields.values
            np.testing.assert_array_equal(context.export(a.fields.values), context.export(b.fields.values))
            old_state, old_stop = first.state, first.stop
            with pytest.raises(ValueError, match='terminal'):
                first.result()
            tree = learner(ops, second)
            train, valid = pair(ops, second, tree)
            second.advance(train, valid)
            second.close()
            assert first.state is old_state and first.stop is old_stop and first.steps == ()
            assert ops._get(a.fields, DeviceFields) is a.fields and ops._get(b.fields, DeviceFields) is b.fields
            first.close()
            # These field records/values are caller-owned even after run closure.
            context.export(a.fields.values)
            for request in (a, b):
                ops.release(request.fields)


@pytest.mark.parametrize('step', ['fixed', 'backtracking'])
def test_finite_prediction_overflow_obeys_trial_and_stop_policy(step):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 1)
            spec = specs[0]

            def constant(o, data, fields):
                value = o.execution.upload(np.array([2], np.float32))
                return device_tree.assemble(o, binning=spec.binning,
                                             topology=((-1, -1, False, -1, -1),), values=(value,))

            options = dict(rounds=1, learning_rate=3e38, step=step, max_trials=3)
            phase = start(ops, spec, **options)
            with _workspace(ops) as retained:
                request = phase.request_tree()
                tree = constant(ops, request.data, request.fields)
                retained.add(tree)
            train, valid = pair(ops, phase, tree)
            state, stop = phase.state, phase.stop
            if step == 'fixed':
                with pytest.raises(ValueError):
                    phase.advance(train, valid)
                assert phase.state is state and phase.stop is stop and phase.steps == ()
                with pytest.raises(ValueError):
                    device_recipes.squared(ops, spec.train, spec.validation, run_id='control', seed=17,
                                           binning=spec.binning, prepared=spec.prepared, learner=constant, **options)
            else:
                actual = phase.advance(train, valid)
                assert not actual.accepted and len(actual.coefficients) == 3 and len(actual.failures) == 1
                assert phase.state is state and phase.stop.completed_rounds == 1
                fit = device_recipes.squared(ops, spec.train, spec.validation, run_id=spec.run_id, seed=17,
                                            binning=spec.binning, prepared=spec.prepared, learner=constant, **options)
                try:
                    assert actual == fit.steps[0] and phase.stop == fit.stop
                    assert phase.result().model.record() == fit.run.export(fit.state).record()
                finally:
                    fit.run.close()
            phase.close()


def test_failed_initialization_releases_all_owned_run_storage(monkeypatch):
    from openboost.device_runtime import DeviceRun

    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 1)
            baseline, records, buffers = context.metrics['live_bytes'], set(ops._records), set(context._buffers)
            original = DeviceRun.initialize

            def fail(run):
                original(run)
                raise MemoryError('declared failure after state initialization')

            with monkeypatch.context() as patch:
                patch.setattr(DeviceRun, 'initialize', fail)
                with pytest.raises(MemoryError, match='declared failure'):
                    start(ops, specs[0])
            assert context.metrics['live_bytes'] == baseline
            assert set(ops._records) == records and set(context._buffers) == buffers


@pytest.mark.parametrize('rate', [8, 64])
def test_multiple_backtracking_trials_match_native_recipe(rate, monkeypatch, tmp_path):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        with _workspace(ops):
            specs, _ = prepare(ops, 1)
            spec = replace(specs[0], options=dict(specs[0].options) | dict(rounds=1, learning_rate=rate, step='backtracking', max_trials=6))
            expected = run_many(ops, (spec,))[0]
            assert expected.result is not None
            assert len(expected.result.steps[0].coefficients) > 1 and expected.result.steps[0].accepted
            with start(ops, spec) as phase:
                tree = learner(ops, phase)
                train, valid = pair(ops, phase, tree)
                with monkeypatch.context() as patch:
                    patch.setattr(device_tree, 'predict', forbidden)
                    phase.advance(train, valid)
                result = phase.result()
                assert record(result) == record(expected.result)
                for item in (train, valid, tree):
                    ops.release(item)
            row = dict(run_id=spec.run_id, seed=spec.seed, options=dict(spec.options), result=record(result),
                       inputs=dict(train=input_snapshot(spec.train), validation=input_snapshot(spec.validation)),
                       predictions={key: {split: getattr(result, key).predict(p.data, offset=p.offset).tolist()
                                          for split, p in (('train', spec.train), ('validation', spec.validation))}
                                    for key in ('model', 'best_model')})
            root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'active-squared'
            root.mkdir(parents=True, exist_ok=True)
            fresh_inference(dict(rate=rate, outcomes=[row]), root / f'backtracking-{rate}.json')
