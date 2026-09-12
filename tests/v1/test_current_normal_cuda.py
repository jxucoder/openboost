"""Complete historical Normal matrix with explicit exact resident tree policies."""
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from openboost import device_newton_order as order
from openboost import device_normal, device_objectives, device_recipes, device_tree
from openboost.device import DeviceOperations
from openboost.device_newton_leaf import leaf
from openboost.execution import ExecutionContext

from .reference.binary32_rounding import bits
from .reference.coupled import normal_scores
from .reference.exact_growth import fit
from .reference.normal_precision import ATOL, LOSS_ATOL, LOSS_RTOL, RTOL
from .test_current_normal_cpu_reference import configuration, cpu_fit, flat, inputs

pytestmark = pytest.mark.gpu


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k:plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def retain(tmp_path, name, record):
    root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'current-normal'
    root.mkdir(parents=True, exist_ok=True)
    (root / (name+'.json')).write_text(json.dumps(plain(record), indent=2, allow_nan=False)+'\n')


def raw(run, state, *, validation=False, best=False):
    handle = run.raw(state, validation=validation, best=best)
    result = run.execution.export(handle)
    run.execution.release(handle)
    return result


def keys(model):
    return [None if f == -1 else (f,t,m) for f,t,m in zip(model['feature'], model['threshold'], model['missing_left'], strict=True)]


def check_record(record, cpu, fitted, expected):
    config = record['configuration']
    _, _, _, _, f = inputs(config)
    assert record['stage'] == 'complete' and record['closed_live_bytes'] == 0 and record['closed_records'] == 0
    assert len(record['trees']) == len(fitted)
    for actual, cpu_tree in zip(record['trees'], fitted, strict=True):
        stored = np.asarray(actual['fields'], np.float32)
        assert stored.view(np.uint32).tolist() == actual['field_bits']
        wanted = fit(f['x'], stored, policy='depthwise', depth=config['depth'],
                     information_minima={2:1,3:1} if config['minimum'] is not None else None)
        assert keys(actual['tree']) == [n['key'] for n in wanted]
        np.testing.assert_array_equal(actual['tree']['left'], [n['left'] for n in wanted])
        np.testing.assert_array_equal(actual['tree']['right'], [n['right'] for n in wanted])
        assert np.asarray(actual['tree']['value'], np.float32).view(np.uint32).reshape(-1).tolist() == [bits(n['value']) for n in wanted]
        # Full current CPU/GPU structural parity is separate from the preceding
        # bit-exact original-device-field oracle, with no approximate tie band.
        assert keys(actual['tree']) == keys(cpu_tree['tree'])
        np.testing.assert_allclose(actual['tree']['value'], cpu_tree['tree']['value'], rtol=RTOL, atol=ATOL)
    cpu_steps = flat(cpu)
    assert len(record['steps']) == len(record['transactions']) == len(record['geometries']) == len(cpu_steps)
    for geometry, host in zip(record['geometries'], cpu_steps, strict=True):
        for name, wanted in [('gradient', host.gradient), ('fisher', host.fisher_diagonal), ('direction', host.direction)]:
            np.testing.assert_allclose(geometry[name], wanted, rtol=RTOL, atol=ATOL)
    for step, transaction, host, ref in zip(record['steps'], record['transactions'], cpu_steps, expected['steps'], strict=True):
        assert (step['round_index'], step['channels'], step['before_version'], step['after_version']) == (
            host.round_index, list(host.channels), host.before_version, host.after_version)
        assert [t['coefficient'] for t in step['trials']] == [float(np.float32(a)) for a in host.coefficients]
        assert [t['accepted'] for t in step['trials']] == [a[1] == 'accepted' for a in ref['attempts']]
        assert [t['failure'] is not None for t in step['trials']] == [a is not None for a in host.failures]
        assert transaction['after_version'] == ref['version'] and transaction['best_terms'] == ref['best_terms']
        for actual, wanted in [(transaction['before'], host.raw_before), (transaction['after'], host.raw_after),
                               (transaction['validation'], ref['validation']), (transaction['best'], ref['best'])]:
            np.testing.assert_allclose(actual, wanted, rtol=RTOL, atol=ATOL)
        assert step['loss'] == pytest.approx(host.loss_after, rel=LOSS_RTOL, abs=LOSS_ATOL)
        assert (step['validation_change'] is None) == (host.validation_change is None)
    assert record['stop']['completed_rounds'] == 3 and record['stop']['reason'] == 'budget'
    assert record['state']['best_n_terms'] == len(cpu.state.best_model.terms) == expected['best_terms']
    for actual, wanted in [(record['raw'], cpu.state.train_raw), (record['validation'], cpu.state.validation_raw),
                           (record['best'], cpu.state.best_model.predict(cpu.state.validation.data))]:
        np.testing.assert_allclose(actual, wanted, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(record['quality'], record['cpu_quality'], rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize('scenario', ['weighted_d1', 'd2_d1', 'd2_constrained', 'conflict_root', 'conflict_d2'])
@pytest.mark.parametrize('geometry', ['ordinary', 'natural', 'damped'])
@pytest.mark.parametrize('update', ['joint', 'forward', 'reverse'])
@pytest.mark.parametrize('step', ['fixed', 'backtracking'])
def test_complete_current_normal_trajectory(scenario, geometry, update, step, tmp_path, monkeypatch):
    config = configuration(scenario, geometry, update, step)
    name = '-'.join((scenario, geometry, update, step))
    train, validation, binned, information, f = inputs(config)
    record = dict(configuration=config, name=name, stage='inputs', trees=[], transactions=[], comparisons=[], geometries=[],
                  inputs=dict(x=f['x'], validation_x=f['validation_x'], binning_cuts=[c.tolist() for c in binned.binning.cuts],
                              train=dict(identity=train.identity, values=[[None if np.isnan(v) else float(v) for v in r] for r in train.data.values],
                                         row_ids=train.data.row_ids.tolist(), target=train.target.tolist(), weight=train.weight.tolist(), offset=train.offset.tolist()),
                              validation=dict(identity=validation.identity, values=[[None if np.isnan(v) else float(v) for v in r] for r in validation.data.values],
                                              row_ids=validation.data.row_ids.tolist(), target=validation.target.tolist(), weight=validation.weight.tolist(), offset=validation.offset.tolist())))
    retain(tmp_path, name, record)
    # The CPU reference is a completed independent cloud prerequisite. Its actual
    # models and arrays are retained as a second trajectory, never fed to CUDA.
    cpu, fitted, expected = cpu_fit(config)
    record['cpu'] = dict(trees=fitted, steps=[asdict(s) for s in flat(cpu)], model=cpu.state.model.record(),
                         best_model=cpu.state.best_model.record(), raw=cpu.state.train_raw, validation=cpu.state.validation_raw)
    record['cpu_quality'] = normal_scores(cpu.state.validation_raw+validation.offset, validation.target[:,0], weight=validation.weight)
    record['stage'] = 'cpu-complete'
    retain(tmp_path, name, record)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        original_compare, original_trials = device_normal.compare, device_recipes.try_terms
        original_direction = device_objectives.diagonal_direction
        def observe_direction(o, gradient, fisher, **options):
            result = original_direction(o, gradient, fisher, **options)
            record['geometries'].append(dict(gradient=context.export(gradient).tolist(), fisher=context.export(fisher).tolist(),
                                             direction=context.export(result).tolist()))
            retain(tmp_path, name, record)
            return result
        def observe_compare(o, problem, before, after):
            result = original_compare(o, problem, before, after)
            # Read-only observation after the policy operation; no host output
            # participates in choosing an update, tree, best prefix or stop.
            record['comparisons'].append(dict(problem=problem.data.problem_identity,
                                              before=context.export(before).tolist(), after=context.export(after).tolist(),
                                              result=asdict(result)))
            retain(tmp_path, name, record)
            return result
        def observe_trials(run, state, terms, **options):
            entry = dict(before_version=state.version, before=raw(run,state).tolist(), stage='before',
                         mappings=[t.mapping.tolist() for t in terms])
            record['transactions'].append(entry)
            retain(tmp_path,name,record)
            updated, trials = original_trials(run,state,terms,**options)
            entry.update(stage='complete', after_version=updated.version, n_terms=updated.n_terms,
                         best_terms=updated.best_n_terms, after=raw(run,updated).tolist(),
                         validation=raw(run,updated,validation=True).tolist(), best=raw(run,updated,validation=True,best=True).tolist(),
                         trials=[asdict(t) for t in trials])
            retain(tmp_path,name,record)
            return updated,trials
        def learner(o,data,fields):
            current=fields
            if config['minimum'] is not None:
                for i, column in enumerate(('cohort:red','cohort:blue')):
                    buffer=context.upload(information[:,i].astype(np.float32))
                    current=o.add_independent(current,column,buffer,nonnegative=True)
                    context.release(buffer)
            stored=context.export(current.values)
            entry=dict(stage='fields',fields=stored.tolist(),field_bits=stored.view(np.uint32).tolist(),names=current.names,roles=current.roles)
            record['trees'].append(entry)
            retain(tmp_path,name,record)
            def choose(o,batch):
                ranked=order.rank(o,batch,min_information={'cohort:red':1,'cohort:blue':1} if config['minimum'] is not None else None)
                return order.choose(o,ranked)
            result=device_tree.depthwise(o,data,current,binning=binned.binning,max_depth=config['depth'],ordering=choose,field_leaf=leaf)
            entry.update(stage='complete',tree=device_tree.export(o,result).record())
            retain(tmp_path,name,record)
            return result
        with monkeypatch.context() as patch:
            patch.setattr(device_normal,'compare',observe_compare)
            patch.setattr(device_recipes,'try_terms',observe_trials)
            patch.setattr(device_objectives,'diagonal_direction',observe_direction)
            result=device_recipes.normal(ops,train,validation,run_id='145-current',seed=7,rounds=3,
                                         binning=binned.binning,learner=learner,mode=config['mode'],damping=config['damping'],
                                         update=config['update'],step=step,learning_rate=config['rate'])
        record.update(stage='fit-complete', steps=[asdict(s) for s in result.steps],stop=dict(asdict(result.stop),reason=result.stop.reason),
                      state=asdict(result.state),model=result.run.export(result.state).record(),best_model=result.run.export(result.state,best=True).record(),
                      raw=raw(result.run,result.state).tolist(),validation=raw(result.run,result.state,validation=True).tolist(),
                      best=raw(result.run,result.state,validation=True,best=True).tolist(),metrics=dict(context.metrics))
        record['quality']=normal_scores(np.asarray(record['validation'])+validation.offset,validation.target[:,0],weight=validation.weight)
        retain(tmp_path,name,record)
        result.run.close()
        record.update(stage='complete',closed_live_bytes=context.metrics['live_bytes'],closed_records=len(ops._records))
        retain(tmp_path,name,record)
    check_record(plain(record),cpu,fitted,expected)
