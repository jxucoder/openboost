"""Host metadata contracts; successful checks do not emulate device execution."""

import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from .test_device_groups_api import jobs as histogram_jobs


def partition_jobs(count=3):
    from openboost.device import DeviceCandidates, DeviceHistogram, DeviceSplit
    from openboost.device_group_tree import PartitionJob
    from openboost.execution import DeviceBuffer

    def buffer(shape, dtype):
        dtype = np.dtype(dtype)
        return DeviceBuffer(shape, dtype.str, int(np.prod(shape)) * dtype.itemsize)

    result = []
    for i, job in enumerate(histogram_jobs(count)):
        hist = DeviceHistogram(job.data, job.fields, job.rows, buffer((2, 5, 2), 'f4'),
                               buffer((2, 5), 'i8'), buffer((2,), 'f4'))
        batch = DeviceCandidates(hist, buffer((16, 2, 2), 'f4'), buffer((16, 2), 'i8'), buffer((16,), 'b1'))
        result.append(PartitionJob(job.run_id, job.rows, DeviceSplit(batch, i % 2)))
    return tuple(result)


def prediction_jobs(count=3, width=2):
    from openboost import NumericData
    from openboost.binning import Binning
    from openboost.device_group_tree import PredictionJob
    from openboost.device_tree import DeviceTree

    binning = Binning.fit(NumericData([[0, 0], [1, 1]], [0, 1], ('x', 'y')), bins=2)
    leaf = (-1, -1, False, -1, -1)
    result = []
    for i, j in enumerate(histogram_jobs(count)):
        topology = (leaf,) if i % 2 == 0 else ((0, 0, True, 1, 2), leaf, leaf)
        data = replace(j.data, binning_identity=binning.identity)
        result.append(PredictionJob(j.run_id, DeviceTree(binning, topology, width), data))
    return tuple(result)


@pytest.mark.parametrize('count', [1, 8, 32])
@pytest.mark.parametrize('width', [1, 2, 3])
def test_node_count_group_count_and_output_width_are_distinct(count, width):
    from openboost.device_group_tree import prediction_plan

    plan = prediction_plan(prediction_jobs(count, width))
    assert plan.shape == (count, 7, width)
    assert plan.node_counts == tuple(1 if i % 2 == 0 else 3 for i in range(count))
    assert plan.packed_bytes == count * (max(plan.node_counts) * (20 + 4 * width) + 1)
    assert plan.output_bytes == count * (7 * width * 4 + 4)
    assert plan.detached_bytes == count * 7 * width * 4
    assert plan.peak_extra_bytes == plan.packed_bytes + plan.output_bytes + plan.detached_bytes


@pytest.mark.parametrize('count', [1, 8, 32])
def test_partition_plan_counts_only_selected_rows_and_compact_metadata(count):
    from openboost.device_group_tree import partition_plan

    plan = partition_plan(partition_jobs(count))
    assert plan.row_counts == tuple(i % 8 for i in range(count))
    assert plan.shape == (count, 2, max(plan.row_counts))
    assert plan.packed_bytes == count * (max(plan.row_counts) * 4 + 17)
    assert plan.output_bytes == count * (max(plan.row_counts) * 8 + 8)
    assert plan.detached_bytes == 4 * sum(plan.row_counts)
    assert plan.peak_extra_bytes == plan.packed_bytes + plan.output_bytes + plan.detached_bytes


@pytest.mark.parametrize('operation', ['partition', 'prediction'])
def test_active_masks_skip_scratch_only_when_all_inactive(operation):
    from openboost import device_group_tree as groups

    jobs = partition_jobs() if operation == 'partition' else prediction_jobs()
    plan = getattr(groups, operation + '_plan')
    assert plan(jobs, active=(False,) * 3).peak_extra_bytes == 0
    mixed = plan(jobs, active=(True, False, True))
    assert mixed.active == (True, False, True)
    assert mixed.detached_bytes == (8 if operation == 'partition' else 112)


@pytest.mark.parametrize('operation', ['partition', 'prediction'])
@pytest.mark.parametrize('fault', ['empty', 'too_many', 'list', 'duplicate', 'id', 'wrong_job',
                                  'mask_length', 'mask_list', 'mask_integer', 'mask_numpy'])
def test_explicit_jobs_and_masks_required(operation, fault):
    from openboost import device_group_tree as groups

    factory = partition_jobs if operation == 'partition' else prediction_jobs
    jobs, kwargs = factory(), {}
    if fault == 'empty':
        jobs = ()
    elif fault == 'too_many':
        jobs = factory(33)
    elif fault == 'list':
        jobs = list(jobs)
    elif fault == 'duplicate':
        jobs = (jobs[0], jobs[0])
    elif fault == 'id':
        jobs = (replace(jobs[0], run_id=''),)
    elif fault == 'wrong_job':
        jobs = histogram_jobs()
    else:
        kwargs['active'] = {'mask_length': (True,), 'mask_list': [True] * 3,
                            'mask_integer': (True, 1, True),
                            'mask_numpy': (True, np.bool_(True), True)}[fault]
    with pytest.raises(ValueError):
        getattr(groups, operation + '_plan')(jobs, **kwargs)


@pytest.mark.parametrize('fault', ['codes', 'missing', 'prepared', 'identity', 'cuts',
                                  'row_data', 'row_dtype', 'row_capacity', 'split_rows',
                                  'split_index', 'split_threshold', 'split_batch'])
def test_partition_incompatible_or_broken_chain_rejected_while_inactive(fault):
    from openboost.device_group_tree import partition_plan

    jobs = partition_jobs()
    j = jobs[1]
    rows, split = j.rows, j.split
    if fault in ('codes', 'missing', 'prepared', 'identity', 'cuts'):
        fields = {'prepared': 'prepared_identity', 'identity': 'data_identity', 'cuts': 'binning_identity'}
        key = fields.get(fault, fault)
        value = replace(getattr(rows.data, key)) if fault in ('codes', 'missing') else 'other'
        data = replace(rows.data, **{key: value})
        rows = replace(rows, data=data)
        hist = replace(split.candidates.histogram, data=data, rows=rows)
        split = replace(split, candidates=replace(split.candidates, histogram=hist))
    elif fault == 'row_data':
        rows = replace(rows, data=jobs[0].rows.data)
    elif fault in ('row_dtype', 'row_capacity'):
        rows = replace(rows, positions=replace(rows.positions, **(
            {'dtype': '<i8'} if fault == 'row_dtype' else {'shape': (8,)})))
        split = replace(split, candidates=replace(split.candidates, histogram=replace(split.candidates.histogram, rows=rows)))
    elif fault == 'split_rows':
        rows = replace(rows)
    else:
        split = replace(split, **{'split_index': {'index': -1}, 'split_threshold': {'index': 4},
                                  'split_batch': {'candidates': object()}}[fault])
    jobs = (jobs[0], replace(j, rows=rows, split=split), jobs[2])
    with pytest.raises(ValueError):
        partition_plan(jobs, active=(True, False, True))


@pytest.mark.parametrize('fault', ['codes', 'missing', 'prepared', 'identity', 'n_rows',
                                  'code_dtype', 'features', 'bins', 'width', 'tree', 'nodes', 'cuts'])
def test_prediction_incompatible_metadata_rejected_while_inactive(fault):
    from openboost.device_group_tree import prediction_plan

    jobs = prediction_jobs()
    j = jobs[1]
    data, tree = j.data, j.tree
    if fault in ('codes', 'missing'):
        data = replace(data, **{fault: replace(getattr(data, fault))})
    elif fault == 'code_dtype':
        data = replace(data, codes=replace(data.codes, dtype='<i8'))
    elif fault in ('prepared', 'identity', 'n_rows', 'features', 'bins', 'cuts'):
        key, value = {'prepared': ('prepared_identity', 'other'), 'identity': ('data_identity', 'other'),
                      'n_rows': ('n_rows', True), 'features': ('feature_names', ('y', 'x')),
                      'bins': ('bin_counts', (0, 2)), 'cuts': ('binning_identity', 'other')}[fault]
        data = replace(data, **{key: value})
    elif fault == 'width':
        tree = replace(tree, output_width=3)
    elif fault == 'nodes':
        tree = replace(tree, topology=())
    else:
        tree = object()
    with pytest.raises(ValueError):
        prediction_plan((jobs[0], replace(j, tree=tree, data=data), jobs[2]), active=(True, False, True))


def test_import_has_no_cuda_or_recipe_dependency():
    code = """
import sys
for name in ('cupy', 'numba', 'openboost.recipes', 'openboost.runs'):
    sys.modules[name] = None
from openboost.device_group_tree import partitions, predictions
assert callable(partitions) and callable(predictions)
"""
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True)


def test_public_grouped_routing_and_prediction_exist():
    from openboost.device_group_tree import (
        DevicePrediction,
        PartitionJob,
        PredictionJob,
        partition_plan,
        partitions,
        prediction_plan,
        predictions,
    )

    assert all(callable(x) for x in (DevicePrediction, PartitionJob, PredictionJob,
                                    partition_plan, partitions, prediction_plan, predictions))
