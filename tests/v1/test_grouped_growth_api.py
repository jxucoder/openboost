"""Host metadata contracts for grouped growth; no CUDA emulation."""

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from .test_device_groups_api import jobs as histogram_jobs


def test_public_grouped_growth_and_assembly_exist():
    from openboost.device_group_growth import TreeJob, TreeOutcome, depthwise, growth_plan
    from openboost.device_tree import assemble

    assert all(callable(x) for x in (TreeJob, TreeOutcome, depthwise, growth_plan, assemble))


def jobs(count=3, width=2, separate=False):
    from openboost.binning import Binning
    from openboost.device_group_growth import TreeJob

    binning = Binning(('x', 'y'), ([0.5], [0.5, 1.5, 2.5]))
    result = []
    for i, j in enumerate(histogram_jobs(count)):
        data = replace(j.data, binning_identity=binning.identity)
        fields = replace(j.fields, data=data)
        leaves = replace(fields, names=('a', 'b')) if separate else None
        result.append(TreeJob(j.run_id, data, fields, binning, max_depth=i % 3,
                              output_width=width, leaf_fields=leaves))
    return tuple(result)


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('width', [1, 2, 4])
@pytest.mark.parametrize('separate', [False, True])
def test_independent_width_depth_and_field_layout(count, width, separate):
    from openboost.device_group_growth import growth_plan

    group = jobs(count, width, separate)
    active = tuple(i % 3 != 1 for i in range(count))
    plan = growth_plan(group, active=active)
    assert plan.run_ids == tuple(j.run_id for j in group)
    assert plan.active == active
    assert plan.depths == tuple(i % 3 for i in range(count))
    assert plan.output_widths == (width,) * count
    assert plan.split_names == ('q0', 'q1')
    assert plan.leaf_names == (('a', 'b') if separate else ('q0', 'q1'))
    with pytest.raises(FrozenInstanceError):
        group[0].max_depth = 3


@pytest.mark.parametrize('fault', ['empty', 'too_many', 'list', 'duplicate', 'id', 'object',
                                  'mask_length', 'mask_list', 'mask_integer', 'mask_numpy'])
def test_explicit_group_and_active_mask(fault):
    from openboost.device_group_growth import growth_plan

    group, kwargs = jobs(), {}
    if fault == 'empty':
        group = ()
    elif fault == 'too_many':
        group = jobs(33)
    elif fault == 'list':
        group = list(group)
    elif fault == 'duplicate':
        group = (group[0], group[0])
    elif fault == 'id':
        group = (replace(group[0], run_id=''),)
    elif fault == 'object':
        group = (object(),)
    else:
        kwargs['active'] = {'mask_length': (True,), 'mask_list': [True] * 3,
                            'mask_integer': (True, 1, True),
                            'mask_numpy': (True, np.bool_(True), True)}[fault]
    with pytest.raises(ValueError):
        growth_plan(group, **kwargs)


@pytest.mark.parametrize('fault', ['codes', 'cuts', 'prepared', 'fields', 'leaf_binding',
                                  'split_names', 'leaf_names', 'unweighted', 'leaf_unweighted',
                                  'dtype', 'leaf_dtype', 'shape', 'leaf_shape'])
def test_incompatible_inputs_rejected_even_when_inactive(fault):
    from openboost.device_group_growth import growth_plan

    group = jobs(separate=True)
    j = group[1]
    if fault in ('codes', 'cuts', 'prepared'):
        key = {'codes': 'codes', 'cuts': 'binning_identity', 'prepared': 'prepared_identity'}[fault]
        data = replace(j.data, **{key: replace(j.data.codes) if fault == 'codes' else 'other'})
        j = replace(j, data=data, fields=replace(j.fields, data=data), leaf_fields=replace(j.leaf_fields, data=data))
    elif fault in ('fields', 'leaf_binding'):
        key = 'fields' if fault == 'fields' else 'leaf_fields'
        j = replace(j, **{key: replace(getattr(j, key), data=group[0].data)})
    else:
        key = 'leaf_fields' if fault.startswith('leaf_') else 'fields'
        fields = getattr(j, key)
        if fault.endswith('names'):
            fields = replace(fields, names=('different', 'order'))
        elif fault.endswith('unweighted'):
            fields = replace(fields, roles=('unweighted', 'training'))
        else:
            fields = replace(fields, values=replace(fields.values, **(
                {'dtype': '<f8'} if fault.endswith('dtype') else {'shape': (7, 3)})))
        j = replace(j, **{key: fields})
    with pytest.raises(ValueError):
        growth_plan((group[0], j, group[2]), active=(True, False, True))


@pytest.mark.parametrize('fault', ['depth', 'bool_depth', 'width', 'bool_width', 'wide', 'nan',
                                  'negative', 'callback', 'scoring_policy', 'leaf_policy', 'mask_policy'])
def test_invalid_or_ambiguous_policy_rejected(fault):
    from openboost.device_group_growth import growth_plan

    j = jobs()[0]
    def callback(*args):
        return None
    changes = {'depth': dict(max_depth=-1), 'bool_depth': dict(max_depth=True),
               'width': dict(output_width=0), 'bool_width': dict(output_width=True),
               'wide': dict(output_width=2**31), 'nan': dict(reg_lambda=np.nan),
               'negative': dict(min_child_h=-1), 'callback': dict(leaf=1),
               'scoring_policy': dict(scoring=callback, split_penalty=1),
               'leaf_policy': dict(leaf=callback, reg_lambda=2),
               'mask_policy': dict(legality=callback, min_child_h=1)}[fault]
    with pytest.raises(ValueError):
        growth_plan((replace(j, **changes),))


def test_import_has_no_cuda_or_recipe_dependency():
    code = """
import sys
for name in ('cupy', 'numba', 'openboost.recipes', 'openboost.device_recipes', 'openboost.runs'):
    sys.modules[name] = None
from openboost.device_group_growth import TreeJob, depthwise, growth_plan
from openboost.device_tree import assemble
assert all(callable(x) for x in (TreeJob, depthwise, growth_plan, assemble))
"""
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True)


@pytest.mark.parametrize('width', [1, 2, 4])
def test_fresh_growth_record_schema_without_device_imports(width, tmp_path):
    from openboost import NumericData
    from openboost.binning import Binning
    from openboost.tree import Tree

    from .grouped_growth_artifacts import FRESH
    from .grouped_tree_artifacts import snapshot
    from .multi_squared_artifacts import fresh_command

    data = NumericData([[0], [np.nan], [2]], [1, 3, 5], ('x',))
    binning = Binning(('x',), ([0.5],))
    model = Tree(binning, [-1], [-1], np.array([False]), [-1], [-1], np.arange(width)[None, :])
    result = dict(run_id='one', model=model.record(), prediction=model.predict(data).tolist())
    path = tmp_path / 'fresh.json'
    path.write_text(json.dumps(dict(input=snapshot(data), schedules=[dict(observed=[result])])) + '\n')
    fresh = subprocess.run(fresh_command(FRESH, str(path)), capture_output=True, text=True, check=True)
    assert json.loads(fresh.stdout) == dict(models=1, predictions_matched=True, training_imports_denied=True)
