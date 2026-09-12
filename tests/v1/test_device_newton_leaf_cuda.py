"""Exact resident leaves: original-row ratios, IEEE output bits and public trees."""

from dataclasses import replace
from fractions import Fraction as F

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.device_newton_leaf import leaf
from openboost.execution import ExecutionContext

from .reference.binary32_rounding import bits
from .reference.normal_split_order import exact_tree, predict
from .test_device_newton_leaf_reference import stored_case
from .test_device_newton_order_cuda import exact_tree_consumer, setup
from .test_device_newton_order_cuda import retain as old_retain
from .test_normal_order_current_reference import topology

pytestmark = pytest.mark.gpu


def retain(tmp_path, name, record):
    old_retain(tmp_path, 'leaf-'+name, record)


def prepare(context, stored):
    data = NumericData(np.zeros((len(stored), 1)), 101+7*np.arange(len(stored)), ('x',))
    p = Problem(data, np.zeros((len(stored), 1)), data.row_ids)
    binned = Binning(('x',), (np.array([]),)).transform(data)
    ops = DeviceOperations(context)
    prepared = ops.prepare(binned, p)
    fields = ops.fields(prepared, context.upload(stored), names=('gradient', 'curvature'), roles=('training', 'training'))
    return ops, prepared, fields


@pytest.mark.parametrize('case', ['zero', 'cancel', 'one', 'negative_one', 'half_even', 'half_above', 'half_below',
                                  'half_odd', 'negative_half_above', 'minimum', 'half_minimum', 'negative_half_minimum',
                                  'above_half_minimum', 'below_half_minimum', 'odd_subnormal_midpoint', 'smallest_normal',
                                  'normal_boundary', 'below_normal_boundary', 'maximum', 'below_overflow', 'overflow',
                                  'above_overflow', 'negative_below_overflow', 'negative_overflow'])
@pytest.mark.parametrize('reverse', [False, True])
def test_original_row_leaf_rounds_once_to_prescribed_binary32_bits(case, reverse, tmp_path):
    stored, regularization, ratio, expected = stored_case(case)
    record = dict(case=case, reverse=reverse, fields=stored.tolist(), field_bits=stored.view(np.uint32).tolist(),
                  regularization=regularization, exact_ratio=str(ratio), expected_bits=expected, stage='inputs')
    name = case+('-reverse' if reverse else '-forward')
    retain(tmp_path, name, record)
    with ExecutionContext() as context:
        ops, data, fields = prepare(context, stored)
        indices = np.arange(len(stored), dtype=np.int32)
        if reverse:
            indices = indices[::-1]
        rows = ops.rows(data, indices)
        handles, records, before = set(context._buffers), set(ops._records), dict(context.metrics)
        if expected is None:
            with pytest.raises(ValueError, match='exact Newton leaf') as error:
                leaf(ops, fields, rows, reg_lambda=regularization)
            record.update(stage='complete', error=str(error.value), output_bits=None)
            retain(tmp_path, name, record)
            assert set(context._buffers) == handles and set(ops._records) == records
            assert context.metrics['live_bytes'] == before['live_bytes']
        else:
            output = leaf(ops, fields, rows, reg_lambda=regularization)
            after = dict(context.metrics)
            actual = int(context.export(output).view(np.uint32)[0])
            record.update(stage='complete', output_bits=actual, before_metrics=before, after_metrics=after)
            retain(tmp_path, name, record)
            assert actual == expected == bits(ratio)
            assert after['upload_bytes'] == before['upload_bytes'] and after['decision_export_bytes'] == before['decision_export_bytes']
            assert after['export_bytes']-before['export_bytes'] == after['validation_export_bytes']-before['validation_export_bytes'] == 8
            assert context.metrics['live_bytes'] == before['live_bytes']+4
            # The scalar output owns its value, independently of source lifetimes.
            ops.release(fields)
            ops.release(rows)
            ops.release(data)
            context.release(fields.values)
            assert int(context.export(output).view(np.uint32)[0]) == expected


@pytest.mark.parametrize('regularization', [0, 1])
def test_empty_original_rows_require_positive_denominator(regularization):
    with ExecutionContext() as context:
        ops, data, fields = prepare(context, np.array([[1, 1]], np.float32))
        rows = ops.rows(data, np.array([], np.int32))
        if regularization:
            assert context.export(leaf(ops, fields, rows)).view(np.uint32).tolist() == [0]
        else:
            with pytest.raises(ValueError, match='exact Newton leaf'):
                leaf(ops, fields, rows, reg_lambda=0)


@pytest.mark.parametrize('fault', ['foreign_fields', 'foreign_rows', 'released_fields', 'released_rows', 'released_values', 'different_data'])
def test_leaf_rejects_stale_and_foreign_bindings_without_work(fault):
    with ExecutionContext() as context:
        ops, data, fields = prepare(context, np.array([[1, 1], [-1, 1]], np.float32))
        rows = ops.rows(data)
        if fault == 'foreign_fields':
            fields = replace(fields)
        elif fault == 'foreign_rows':
            rows = replace(rows)
        elif fault == 'released_fields':
            ops.release(fields)
        elif fault == 'released_rows':
            ops.release(rows)
        elif fault == 'released_values':
            context.release(fields.values)
        else:
            other = NumericData([[0], [0]], [301, 308], ('x',))
            problem = Problem(other, [[0], [0]], other.row_ids)
            other_data = ops.prepare(Binning(('x',), (np.array([]),)).transform(other), problem)
            rows = ops.rows(other_data)
        before, handles, records = dict(context.metrics), set(context._buffers), set(ops._records)
        with pytest.raises(ValueError):
            leaf(ops, fields, rows)
        assert set(context._buffers) == handles and set(ops._records) == records
        assert context.metrics['kernel_launches'] == before['kernel_launches']
        assert context.metrics['live_bytes'] == before['live_bytes']


@pytest.mark.parametrize('index', [1, 2, 3])
def test_partial_leaf_allocation_failure_preserves_input_and_allows_explicit_retry(index, monkeypatch):
    with ExecutionContext() as context:
        ops, data, fields = prepare(context, stored_case('cancel')[0])
        rows = ops.rows(data)
        handles, records, live = set(context._buffers), set(ops._records), context.metrics['live_bytes']
        original, calls = context._empty, 0
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == index:
                    raise MemoryError('injected leaf allocation')
                return original(*args, **kwargs)
            patch.setattr(context, '_empty', fail)
            with pytest.raises(MemoryError, match='injected leaf'):
                leaf(ops, fields, rows)
        assert set(context._buffers) == handles and set(ops._records) == records and context.metrics['live_bytes'] == live
        assert context.export(leaf(ops, fields, rows)).view(np.uint32).tolist() == [0xbe800000]


@pytest.mark.parametrize('case', ['balanced', 'cancelled'])
def test_exact_sum_can_exceed_float32_without_a_floating_histogram(case, tmp_path, monkeypatch):
    maximum = np.finfo(np.float32).max
    stored = np.array([[maximum, maximum], [maximum, maximum]], np.float32) if case == 'balanced' else np.array(
        [[maximum, 1], [maximum, 1], [-maximum, 1], [-maximum, 1], [1, 1]], np.float32)
    ratio = F(-1) if case == 'balanced' else F(-1, 6)
    regularization = 0 if case == 'balanced' else 1
    record = dict(case=case, fields=stored.tolist(), regularization=regularization, exact_ratio=str(ratio), stage='inputs')
    retain(tmp_path, 'large-'+case, record)
    with ExecutionContext() as context:
        ops, data, fields = prepare(context, stored)
        rows = ops.rows(data)
        def forbidden(*args, **kwargs):
            raise AssertionError('exact original-row leaf must not construct a floating histogram')
        monkeypatch.setattr(ops, 'histogram', forbidden)
        actual = int(context.export(leaf(ops, fields, rows, reg_lambda=regularization)).view(np.uint32)[0])
        record.update(stage='complete', output_bits=actual)
        retain(tmp_path, 'large-'+case, record)
        assert actual == bits(ratio)


@pytest.mark.parametrize('case', ['captured', 'p24-negative', 'p24-zero', 'p24-positive',
                                  'p54-negative', 'p54-zero', 'p54-positive',
                                  'p100-negative', 'p100-zero', 'p100-positive',
                                  'p149-negative', 'p149-zero', 'p149-positive'])
def test_public_exact_order_tree_composes_with_original_row_leaf(case, tmp_path, monkeypatch):
    with ExecutionContext() as context:
        ops, data, fields, problem, binned, x, exact = setup(context, case)
        record = dict(case=case, stage='inputs', x=x, row_ids=problem.data.row_ids.tolist(),
                      fields=[[float(v) for v in row] for row in exact])
        retain(tmp_path, 'tree-'+case, record)
        calls = []
        def exact_leaf(histogram):
            calls.append(histogram.rows)
            return leaf(ops, histogram.fields, histogram.rows)
        monkeypatch.setattr(ops, 'leaf', exact_leaf)
        tree = exact_tree_consumer(ops, data, fields, binned)
        exported = trees.export(ops, tree)
        prediction = context.export(trees.predict(ops, tree, data))
        record.update(stage='complete', tree=exported.record(), prediction=prediction.tolist())
        retain(tmp_path, 'tree-'+case, record)
        wanted = exact_tree(x, exact, depth=2)
        assert len(calls) == len(wanted) and list(tree.topology) == topology(wanted)
        actual_bits = exported.value.astype(np.float32).view(np.uint32).reshape(-1).tolist()
        assert actual_bits == [bits(n['value']) for n in wanted]
        assert prediction.view(np.uint32).reshape(-1).tolist() == [bits(v) for v in predict(wanted, x)]


@pytest.mark.parametrize('fault', ['negative_curvature', 'unweighted', 'negative_lambda', 'infinite_lambda', 'boolean_lambda', 'overflow_lambda'])
def test_invalid_leaf_domain_or_policy_preserves_owned_inputs(fault):
    with ExecutionContext() as context:
        stored = np.array([[1, -1 if fault == 'negative_curvature' else 1]], np.float32)
        ops, data, fields = prepare(context, stored)
        if fault == 'unweighted':
            fields = ops.fields(data, fields.values, names=fields.names, roles=('unweighted', 'unweighted'))
        rows = ops.rows(data)
        regularization = dict(negative_lambda=-1, infinite_lambda=np.inf, boolean_lambda=True, overflow_lambda=1e100).get(fault, 1)
        handles, records, live = set(context._buffers), set(ops._records), context.metrics['live_bytes']
        with pytest.raises(ValueError):
            leaf(ops, fields, rows, reg_lambda=regularization)
        assert set(context._buffers) == handles and set(ops._records) == records and context.metrics['live_bytes'] == live
