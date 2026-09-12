"""Host grouping contracts; metadata records cannot execute or emulate CUDA."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest


def test_public_grouped_histogram_operations_exist():
    from openboost.device_groups import HistogramJob, HistogramOutcome, histogram_plan, histograms

    assert all(callable(x) for x in (HistogramJob, HistogramOutcome, histogram_plan, histograms))


def jobs(count=3, *, width=2):
    from openboost.device import DeviceData, DeviceFields, DeviceRows
    from openboost.device_groups import HistogramJob
    from openboost.execution import DeviceBuffer

    def buffer(shape, dtype):
        dtype = np.dtype(dtype)
        return DeviceBuffer(shape, dtype.str, int(np.prod(shape)) * dtype.itemsize)

    codes, missing = buffer((2, 7), 'i4'), buffer((2, 7), 'b1')
    result = []
    for i in range(count):
        data = DeviceData('data', 'prepared', f'problem-{i}', 'cuts', 7,
                          ('x', 'y'), (2, 4), codes, missing, buffer((7,), 'f4'))
        fields = DeviceFields(data, tuple(f'q{j}' for j in range(width)),
                              ('training',) * width, buffer((7, width), 'f4'))
        rows = DeviceRows(data, buffer((i % 8,), 'i4'))
        result.append(HistogramJob(f'run-{i}', data, fields, rows))
    return tuple(result)


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('width', [1, 2, 5])
def test_group_axis_is_separate_from_field_width_and_original_row_counts(count, width):
    from openboost.device_groups import histogram_plan

    group = jobs(count, width=width)
    plan = histogram_plan(group)
    assert plan.run_ids == tuple(j.run_id for j in group)
    assert plan.shape == (count, 2, 5, width)
    assert plan.row_counts == tuple(i % 8 for i in range(count))
    assert plan.active == (True,) * count
    assert plan.packed_bytes == count * (7 * width * 4 + max(plan.row_counts) * 4 + 5)
    one = 2 * 5 * width * 4 + 2 * 5 * 8 + width * 4
    assert plan.reduction_bytes == count * (one + 4)
    assert plan.detached_bytes == count * one
    assert plan.peak_extra_bytes == plan.packed_bytes + plan.reduction_bytes + plan.detached_bytes


def test_inactive_slots_are_explicit_and_all_inactive_needs_no_scratch():
    from openboost.device_groups import histogram_plan

    group = jobs(3)
    plan = histogram_plan(group, active=(True, False, True))
    assert plan.detached_bytes == 2 * (2 * 5 * 2 * 4 + 2 * 5 * 8 + 2 * 4)
    assert histogram_plan(group, active=(False,) * 3).peak_extra_bytes == 0


@pytest.mark.parametrize('fault', ['empty', 'too_many', 'list', 'duplicate', 'id', 'object',
                                  'mask_length', 'mask_list', 'mask_integer', 'mask_numpy'])
def test_group_and_mask_must_be_explicit(fault):
    from openboost.device_groups import histogram_plan

    group, kwargs = jobs(3), {}
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
        histogram_plan(group, **kwargs)


@pytest.mark.parametrize('fault', ['codes', 'missing', 'cuts', 'prepared', 'identity', 'rows',
                                  'features', 'bins', 'names', 'roles', 'unweighted',
                                  'field_data', 'row_data', 'field_dtype', 'field_shape',
                                  'row_dtype', 'row_shape', 'row_capacity'])
def test_incompatible_group_is_rejected_even_when_slot_is_inactive(fault):
    from openboost.device_groups import histogram_plan

    group = jobs(3)
    j = group[1]
    data, fields, rows = j.data, j.fields, j.rows
    if fault in ('codes', 'missing'):
        data = replace(data, **{fault: replace(getattr(data, fault))})
    elif fault in ('cuts', 'prepared', 'identity', 'rows', 'features', 'bins'):
        key, value = {'cuts': ('binning_identity', 'other'), 'prepared': ('prepared_identity', 'other'),
                      'identity': ('data_identity', 'other'), 'rows': ('n_rows', 6),
                      'features': ('feature_names', ('y', 'x')), 'bins': ('bin_counts', (4, 2))}[fault]
        data = replace(data, **{key: value})
    elif fault in ('names', 'roles', 'unweighted'):
        key, value = {'names': ('names', ('q1', 'q0')), 'roles': ('roles', ('independent',) * 2),
                      'unweighted': ('roles', ('unweighted',) * 2)}[fault]
        fields = replace(fields, **{key: value})
    elif fault == 'field_data':
        fields = replace(fields, data=group[0].data)
    elif fault == 'row_data':
        rows = replace(rows, data=group[0].data)
    elif fault.startswith('field_'):
        fields = replace(fields, values=replace(fields.values, **(
            {'dtype': '<f8'} if fault == 'field_dtype' else {'shape': (7, 3)})))
    else:
        rows = replace(rows, positions=replace(rows.positions, **{
            'row_dtype': {'dtype': '<i8'}, 'row_shape': {'shape': (1, 1)},
            'row_capacity': {'shape': (8,)}}[fault]))
    if data is not j.data:
        fields, rows = replace(fields, data=data), replace(rows, data=data)
    group = (group[0], replace(j, data=data, fields=fields, rows=rows), group[2])
    with pytest.raises(ValueError):
        histogram_plan(group, active=(True, False, True))


def test_import_is_cuda_and_training_independent():
    code = """
import sys
for name in ('cupy', 'numba', 'openboost.recipes', 'openboost.runs'):
    sys.modules[name] = None
from openboost.device_groups import HistogramJob, histogram_plan, histograms
assert callable(histograms)
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)
