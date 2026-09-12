"""137 real-device consumers of grouped predictions; no scheduler/speed claim."""

import json
import os
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from openboost import device_inputs, device_multi_squared, device_objectives, device_tree
from openboost.device import DeviceOperations, _workspace
from openboost.device_group_tree import PredictionJob, predictions
from openboost.device_runtime import DeviceRun, DeviceTerm, PredictedTerm
from openboost.execution import ExecutionContext
from openboost.stopping import StopState

from .multi_squared_artifacts import fresh_command, input_snapshot
from .reference import multi_squared as ref
from .test_device_runtime_cuda import raw
from .test_multi_squared_reference import prepared

pytestmark = pytest.mark.gpu


FRESH = """
import sys
for name in ('openboost.device_group_tree', 'openboost.device_runtime',
             'openboost.device_tree', 'openboost.device_runs', 'openboost.device_inputs'):
    sys.modules[name] = None
import base64, json
from pathlib import Path
import numpy as np
from openboost import NumericData
from openboost.artifacts import Model
def unpack(row):
    return np.frombuffer(base64.b64decode(row['data_base64']),dtype=row['dtype']).reshape(row['shape'])
record=json.loads(Path(sys.argv[1]).read_text())
models=0
for run in record['runs']:
    for key in ('model','best_model'):
        model=Model.from_record(run[key])
        for split in ('train','validation'):
            source=run['inputs'][split]
            data=NumericData(unpack(source['features']),unpack(source['row_ids']),model.feature_names)
            actual=model.predict(data,offset=unpack(source['offset']))
            np.testing.assert_array_equal(actual,run['predictions'][key][split])
        models+=1
print(json.dumps(dict(models=models, arrays=models*2, training_imports_denied=True)))
"""


def setup(ops, count=1, width=2, comparison='objective'):
    train, validation, binning = prepared(width)
    features = tuple(device_inputs.prepare(ops, binning.transform(p.data)) for p in (train, validation))
    runs, hosts = [], []
    for i in range(count):
        host = (
            replace(train, target=train.target + (i % 3)/8, weight=train.weight * 2**(i % 2)),
            replace(validation, target=validation.target + (i % 5)/16),
        )
        runs.append(DeviceRun(ops, *host, run_id=f'predicted-{i}', seed=31, binning=binning,
                              objective=device_multi_squared.objective(), comparison=comparison, prepared=features))
        hosts.append(host)
    return runs, hosts, features


def tree(run, state, depth=1):
    ops = run.ops
    with _workspace(ops) as retained:
        current = run.raw(state)
        g, h = device_multi_squared.geometry(ops, run.problem, current)
        fields = device_objectives.vector_fields(ops, run.data, g, h)
        original = device_tree.depthwise(
            ops, run.data, fields, binning=run.binning, max_depth=depth, output_width=run.raw_width,
            scoring=lambda o, b: o.vector_scores(b), legality=lambda o, b: o.vector_feasible(b),
            leaf=lambda o, hist: o.vector_leaf(hist),
        )
        snapshot = device_tree.copy(ops, original)
        retained.add(snapshot)
        return snapshot


def grouped(ops, runs, trees, indices):
    a = predictions(ops, tuple(PredictionJob(runs[i].run_id, trees[i], runs[i].data) for i in indices))
    b = predictions(ops, tuple(PredictionJob(runs[i].run_id, trees[i], runs[i].validation_data) for i in indices))
    assert all(x.status == 'complete' for x in (*a, *b))
    width = trees[indices[0]].output_width
    mapping = np.eye(width, dtype=np.float32)*np.float32(-.25)
    if width > 1:
        mapping[0, 1] = .125
    return {i: (PredictedTerm(DeviceTerm(trees[i], np.eye(width)), x.prediction, y.prediction),
                PredictedTerm(DeviceTerm(trees[i], mapping), x.prediction, y.prediction))
            for i, x, y in zip(indices, a, b, strict=True)}


def release_predictions(ops, items):
    for record in {p for item in items for p in (item.train, item.validation)}:
        ops.release(record)


def forbid(*args, **kwargs):
    pytest.fail('ordinary tree traversal called while consuming registered predictions')


def arrays(context, run, state):
    return [raw(context, run, state, validation=split) for split in (False, True)]


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('width,comparison', [(1, 'reported'), (2, 'objective')])
def test_two_round_grouped_proposals_match_independent_terms(count, width, comparison, monkeypatch, tmp_path):
    with ExecutionContext(max_bytes=32*1024**2) as context:
        ops = DeviceOperations(context)
        runs, hosts, features = setup(ops, count, width, comparison)
        states = [r.initialize() for r in runs]
        stops = [StopState.start(s.validation_score, rounds=2) for s in states]
        history = [[] for _ in runs]
        schedules = [x for x in (list(range(count)), list(reversed(range(count))),
                                 list(range(0, count, 2)), list(range(1, count, 2))) if x]
        for iteration in range(2):
            trees = [tree(r, s, i % 2) for i, (r, s) in enumerate(zip(runs, states, strict=True))]
            next_states = {}
            for schedule_index, indices in enumerate(schedules):
                items = grouped(ops, runs, trees, indices)
                for i in indices:
                    run, state = runs[i], states[i]
                    coefficient = np.float32((.5, 0., -.25)[i % 3])
                    accept = (i + iteration) % 4 != 1
                    before = arrays(context, run, state)
                    # Independent existing transaction is outside the traversal guard.
                    control = run.propose_terms(state, tuple(t.term for t in items[i]), coefficient=coefficient)
                    wanted = arrays(context, run, control)
                    control_state = run.resolve(state, control, accept=accept)
                    expected_models = [run.export(control_state, best=b).record() for b in (False, True)]
                    expected_state = asdict(control_state)
                    run.release(control)
                    if control_state is not state:
                        run.release(control_state)
                    predicted_values = [context.export(p.values) for p in (items[i][0].train, items[i][0].validation)]
                    exact = [a.copy() for a in before]
                    for item in items[i]:
                        exact = [ref.mapped(a, p, item.term.mapping, coefficient)
                                 for a, p in zip(exact, predicted_values, strict=True)]
                    with monkeypatch.context() as guard:
                        guard.setattr(device_tree, 'predict', forbid)
                        metrics = dict(context.metrics)
                        proposal = run.propose_predicted(state, items[i], coefficient=coefficient)
                        assert context.metrics['upload_bytes'] == metrics['upload_bytes']
                        updated = run.resolve(state, proposal, accept=accept)
                    for observed, previous, expected in zip(arrays(context, run, proposal), wanted, exact, strict=True):
                        np.testing.assert_array_equal(observed, previous)
                        np.testing.assert_array_equal(observed, expected)
                    for a, b in zip(arrays(context, run, state), before, strict=True):
                        np.testing.assert_array_equal(a, b)
                    assert [run.export(updated, best=b).record() for b in (False, True)] == expected_models
                    actual_state = asdict(updated)
                    assert {k: v for k, v in actual_state.items() if k != 'identity'} == {
                        k: v for k, v in expected_state.items() if k != 'identity'}
                    row = dict(round=iteration, coefficient=float(coefficient), accepted=accept,
                               before=[x.tolist() for x in before], prediction=[x.tolist() for x in predicted_values],
                               mappings=[t.term.mapping.tolist() for t in items[i]],
                               proposed=[x.tolist() for x in wanted], tree=device_tree.export(ops, trees[i]).record(),
                               after=[x.tolist() for x in arrays(context, run, updated)], state=actual_state)
                    run.release(proposal)
                    if schedule_index == 0:
                        next_states[i] = updated
                        history[i].append(row)
                    elif updated is not state:
                        run.release(updated)
                    release_predictions(ops, items[i])
            for i, run in enumerate(runs):
                ops.release(trees[i])
                if states[i] is not next_states[i]:
                    run.release(states[i])
                states[i] = next_states[i]
                stops[i] = stops[i].observe(states[i].validation_score)
        report = dict(count=count, width=width, comparison=comparison, schedules=schedules, runs=[])
        for i, (run, state, host) in enumerate(zip(runs, states, hosts, strict=True)):
            models = {key: run.export(state, best=best) for key, best in (('model', False), ('best_model', True))}
            report['runs'].append(dict(
                run_id=run.run_id, seed=run.seed, history=history[i], stop=asdict(stops[i]),
                rng=run.rng(2, 'learner', 'rows').integers(0, 2**32, 8).tolist(),
                inputs={k: input_snapshot(p) for k, p in zip(('train', 'validation'), host, strict=True)},
                **{k: m.record() for k, m in models.items()},
                predictions={k: {s: m.predict(p.data, offset=p.offset).tolist()
                                 for s, p in zip(('train', 'validation'), host, strict=True)} for k, m in models.items()},
            ))
            run.close()
        for feature in features:
            ops.release(feature)
        assert context.metrics['live_bytes'] == 0
    root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'predicted-proposals'
    root.mkdir(parents=True, exist_ok=True)
    path = root / f'M{count}-K{width}.json'
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    fresh = subprocess.run(fresh_command(FRESH, str(path)), capture_output=True, text=True, check=True)
    report['fresh_inference'] = json.loads(fresh.stdout)
    assert report['fresh_inference'] == dict(models=2*count, arrays=4*count, training_imports_denied=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


@pytest.mark.parametrize('fault', ['forged_prediction', 'released_prediction', 'released_values',
                                  'released_tree', 'foreign_binding', 'swapped_splits',
                                  'foreign_state', 'forged_state', 'closed_run'])
def test_live_ownership_is_checked_before_candidate_allocation(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        runs, _, _ = setup(ops, 2)
        states = [r.initialize() for r in runs]
        trees = [tree(r, s, 0) for r, s in zip(runs, states, strict=True)]
        item = grouped(ops, runs, trees, [0, 1])
        run, state, values = runs[0], states[0], item[0]
        if fault == 'forged_prediction':
            values = (replace(values[0], train=replace(values[0].train)),)
        elif fault == 'released_prediction':
            ops.release(values[0].train)
        elif fault == 'released_values':
            context.release(values[0].validation.values)
        elif fault == 'released_tree':
            ops.release(trees[0])
        elif fault == 'foreign_binding':
            other = predictions(ops, (PredictionJob('other', trees[0], runs[1].data),))[0].prediction
            values = (replace(values[0], train=other),)
        elif fault == 'swapped_splits':
            values = (replace(values[0], train=values[0].validation, validation=values[0].train),)
        elif fault == 'foreign_state':
            state = states[1]
        elif fault == 'forged_state':
            state = replace(state)
        else:
            run.close()
        buffers, records, serial = set(context._buffers), set(ops._records), run._serial
        with pytest.raises((ValueError, RuntimeError)):
            run.propose_predicted(state, values)
        assert buffers == set(context._buffers) and records == set(ops._records) and serial == run._serial
        # Neighbor predictions and accepted state remain usable even after a failed slot.
        neighbor = runs[1].propose_predicted(states[1], item[1], coefficient=.5)
        runs[1].release(neighbor)


@pytest.mark.parametrize('failure', [1, 3, 5, 7, 11])
def test_partial_allocation_failure_preserves_inputs_and_same_id_retry(failure, monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        runs, _, _ = setup(ops)
        run = runs[0]
        state = run.initialize()
        items = grouped(ops, runs, [tree(run, state, 0)], [0])[0]
        previous = arrays(context, run, state)
        rng = run.rng(0, 'tree', 'rows').integers(0, 1000, 8)
        buffers, records, serial = set(context._buffers), set(ops._records), run._serial
        original, calls = context._empty, 0

        def allocation(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == failure:
                raise MemoryError('137 planned allocation failure')
            return original(*args, **kwargs)

        with monkeypatch.context() as guard:
            guard.setattr(context, '_empty', allocation)
            with pytest.raises(MemoryError, match='planned allocation'):
                run.propose_predicted(state, items, coefficient=.5)
        assert calls == failure
        assert buffers == set(context._buffers) and records == set(ops._records) and serial == run._serial
        for a, b in zip(arrays(context, run, state), previous, strict=True):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(run.rng(0, 'tree', 'rows').integers(0, 1000, 8), rng)
        with monkeypatch.context() as guard:
            guard.setattr(device_tree, 'predict', forbid)
            rejected = run.propose_predicted(state, items, coefficient=.5)
            assert run.resolve(state, rejected, accept=False) is state
            expected = arrays(context, run, rejected)
            run.release(rejected)
            retried = run.propose_predicted(state, items, coefficient=.5)
        for a, b in zip(arrays(context, run, retried), expected, strict=True):
            np.testing.assert_array_equal(a, b)
        release_predictions(ops, items)
        ops.release(items[0].term.tree)
        accepted = run.resolve(state, retried, accept=True)
        run.release(retried)
        run.release(state)
        assert len(run.export(accepted).terms) == 2


@pytest.mark.parametrize('fault', ['empty', 'list', 'bare_term', 'raw_buffer', 'nan', 'bool', 'width'])
def test_invalid_predicted_transaction_inputs_do_not_allocate(fault):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        runs, _, _ = setup(ops)
        run, coefficient = runs[0], .5
        state = run.initialize()
        items = grouped(ops, runs, [tree(run, state, 0)], [0])[0]
        if fault == 'empty':
            items = ()
        elif fault == 'list':
            items = list(items)
        elif fault == 'bare_term':
            items = (items[0].term,)
        elif fault == 'raw_buffer':
            items = (items[0].train.values,)
        elif fault == 'width':
            items = (replace(items[0], term=DeviceTerm(items[0].term.tree, np.ones((2, 3)))),)
        else:
            coefficient = float('nan') if fault == 'nan' else True
        buffers, records, serial = set(context._buffers), set(ops._records), run._serial
        with pytest.raises(ValueError):
            run.propose_predicted(state, items, coefficient=coefficient)
        assert buffers == set(context._buffers) and records == set(ops._records) and serial == run._serial


def test_finite_mapping_overflow_is_atomic_and_predictions_can_be_reused(monkeypatch):
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        runs, _, _ = setup(ops)
        run = runs[0]
        state = run.initialize()
        with _workspace(ops) as retained:
            current = run.raw(state)
            g, h = device_multi_squared.geometry(ops, run.problem, current)
            fields = device_objectives.vector_fields(ops, run.data, g, h)
            constant = device_tree.depthwise(
                ops, run.data, fields, binning=run.binning, max_depth=0, output_width=2,
                leaf=lambda o, h: o.execution.upload(np.array([2, 2], np.float32)))
            retained.add(constant)
        items = grouped(ops, runs, [constant], [0])[0]
        bad = (replace(items[0], term=DeviceTerm(constant, np.eye(2, dtype=np.float32)*np.float32(3e38))),)
        buffers, records, serial = set(context._buffers), set(ops._records), run._serial
        with monkeypatch.context() as guard:
            guard.setattr(device_tree, 'predict', forbid)
            with pytest.raises(ValueError, match='finite'):
                run.propose_predicted(state, bad)
            assert buffers == set(context._buffers) and records == set(ops._records) and serial == run._serial
            good = run.propose_predicted(state, items, coefficient=.25)
        assert np.isfinite(raw(context, run, good)).all()
