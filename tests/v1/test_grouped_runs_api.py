"""Host schedule contracts only; all execution is gated on Modal."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import pytest

from openboost import device_recipes
from openboost.device_group_runs import run_many, schedule_plan

from .test_device_inputs_api import pair
from .test_device_runs_reference import jobs


def prepared(count=3):
    specs = jobs(count)
    features = pair(specs[0])
    return tuple(replace(s, prepared=features) for s in specs)


def test_public_schedule_and_execution_api_exists():
    assert callable(schedule_plan) and callable(run_many)


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('size', [1, 3, 32])
def test_plan_owns_immutable_caller_order_groups_and_independent_policies(count, size):
    specs = prepared(count)
    plan = schedule_plan(specs, group_size=size)
    ids = tuple(s.run_id for s in specs)
    assert plan.run_ids == ids
    assert plan.groups == tuple(ids[i:i+size] for i in range(0, count, size))
    assert len(plan.configurations) == count
    assert [p.rounds for p in plan.configurations] == [s.options['rounds'] for s in specs]
    assert [p.patience for p in plan.configurations] == [s.options['patience'] for s in specs]
    assert len({id(p) for p in plan.configurations}) == count
    with pytest.raises(FrozenInstanceError):
        plan.run_ids = ()
    assert schedule_plan(reversed(specs), group_size=size).run_ids == ids[::-1]


@pytest.mark.parametrize('fault', ['empty', 'large', 'object', 'duplicate', 'recipe', 'missing_pair',
                                  'copied_train', 'copied_validation', 'foreign_data', 'cuts',
                                  'unknown_option', 'learner', 'bins', 'rounds', 'trials'])
def test_unsupported_group_rejects_before_device_access(fault):
    specs = prepared()
    if fault == 'empty':
        specs = ()
    elif fault == 'large':
        specs = prepared(33)
    elif fault == 'object':
        specs = (object(),)
    elif fault == 'duplicate':
        specs = (specs[0], replace(specs[0], seed=99))
    else:
        bad = specs[1]
        if fault == 'recipe':
            bad = replace(bad, recipe=lambda *a, **k: None)
        elif fault == 'missing_pair':
            bad = replace(bad, prepared=None)
        elif fault in ('copied_train', 'copied_validation'):
            features = list(bad.prepared)
            index = int(fault == 'copied_validation')
            features[index] = replace(features[index])
            bad = replace(bad, prepared=tuple(features))
        elif fault == 'foreign_data':
            from openboost import NumericData

            data = bad.train.data
            other = NumericData(data.values, data.row_ids+1, data.feature_names)
            bad = replace(bad, train=replace(bad.train, data=other, row_ids=other.row_ids))
        elif fault == 'cuts':
            from openboost.binning import Binning

            bad = replace(bad, binning=Binning.fit(bad.train.data, bins=2))
        else:
            options = dict(bad.options)
            options[{'unknown_option': 'unknown', 'learner': 'learner', 'bins': 'bins',
                     'rounds': 'rounds', 'trials': 'max_trials'}[fault]] = {'rounds': -1, 'trials': 7}.get(fault, 1)
            bad = replace(bad, options=options)
        specs = (specs[0], bad, specs[2])
    with pytest.raises(ValueError):
        schedule_plan(specs)
    with pytest.raises(ValueError):
        run_many(None, specs)


@pytest.mark.parametrize('fault', ['zero', 'negative', 'large', 'boolean', 'float', 'none'])
def test_invalid_group_size_rejects_before_device_access(fault):
    size = dict(zero=0, negative=-1, large=33, boolean=True, float=1.5, none=None)[fault]
    with pytest.raises(ValueError):
        run_many(None, prepared(), group_size=size)


def test_same_shared_records_accept_distinct_pair_tuple_and_inferred_cuts():
    specs = prepared()
    second = replace(specs[1], prepared=tuple(list(specs[0].prepared)), binning=None)
    assert schedule_plan((specs[0], second)).run_ids == ('job-0', 'job-1')
    assert second.prepared is not specs[0].prepared
    assert second.recipe is device_recipes.squared
    with pytest.raises(ValueError, match='DeviceOperations'):
        run_many(None, (specs[0], second))


def test_import_has_no_cuda_or_cpu_trainer_dependency():
    code = """
import sys
for name in ('cupy','numba','openboost.recipes','openboost.runs'):
    sys.modules[name] = None
from openboost.device_group_runs import run_many, schedule_plan
assert callable(run_many) and callable(schedule_plan)
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)
