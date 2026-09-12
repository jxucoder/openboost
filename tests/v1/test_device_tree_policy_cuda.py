"""Standard builder consumes explicit exact policies; original GPU defaults retained."""
from dataclasses import replace

import numpy as np
import pytest

from openboost import device_newton_order as order
from openboost import device_tree as trees
from openboost.device_newton_leaf import leaf
from openboost.execution import ExecutionContext

from .reference.binary32_rounding import bits
from .reference.normal_split_order import predict
from .reference.tree_numerical_policy import tree as reference
from .test_device_newton_leaf_cuda import prepare
from .test_device_newton_order_cuda import retain, setup
from .test_normal_order_current_reference import topology

pytestmark = pytest.mark.gpu


def exact_order(ops, batch, **policy):
    return order.choose(ops, order.rank(ops, batch, **policy))


def verify_tree(context, ops, data, tree, record, tmp_path, name):
    exported = trees.export(ops, tree)
    prediction_buffer = trees.predict(ops, tree, data)
    prediction = context.export(prediction_buffer)
    context.release(prediction_buffer)
    record.update(stage='complete', tree=exported.record(), prediction=prediction.tolist())
    retain(tmp_path, 'policy-'+name, record)
    wanted = reference(record['x'], record['fields'], record.get('leaf_fields'), **record.get('parameters', {}))
    assert list(tree.topology) == topology(wanted)
    assert exported.value.astype(np.float32).view(np.uint32).reshape(-1).tolist() == [bits(n['value']) for n in wanted]
    assert prediction.view(np.uint32).reshape(-1).tolist() == [bits(v) for v in predict(wanted, record['x'])]
    return exported


@pytest.mark.parametrize('case', ['captured', 'p24-negative', 'p24-zero', 'p24-positive',
                                  'p54-negative', 'p54-zero', 'p54-positive',
                                  'p100-negative', 'p100-zero', 'p100-positive',
                                  'p149-negative', 'p149-zero', 'p149-positive'])
def test_standard_builder_exact_tree_and_owned_callback_workspace(case, tmp_path):
    with ExecutionContext() as context:
        ops, data, fields, problem, binned, x, exact = setup(context, case)
        record = dict(case=case, kind='exact', stage='inputs', x=x, fields=[[float(v) for v in r] for r in exact], row_ids=problem.data.row_ids.tolist())
        retain(tmp_path, 'policy-'+case, record)
        buffers, records = set(context._buffers), set(ops._records)
        result = trees.depthwise(ops, data, fields, binning=binned.binning, ordering=exact_order, field_leaf=leaf)
        assert set(ops._records)-records == {result}
        assert set(context._buffers)-buffers == set(ops._records[result][0])
        exported = verify_tree(context, ops, data, result, record, tmp_path, case)
        ops.release(fields)
        ops.release(data)
        context.release(fields.values)
        assert trees.export(ops, result).record() == exported.record()


@pytest.mark.parametrize('case', ['balanced', 'cancelled'])
def test_root_only_field_leaf_avoids_overflowing_histogram(case, tmp_path, monkeypatch):
    maximum = np.finfo(np.float32).max
    stored = np.array([[maximum, maximum], [maximum, maximum]], np.float32) if case == 'balanced' else np.array(
        [[maximum, 1], [maximum, 1], [-maximum, 1], [-maximum, 1], [1, 1]], np.float32)
    regularization = 0 if case == 'balanced' else 1
    from openboost.binning import Binning
    binning = Binning(('x',), (np.array([]),))
    record = dict(case=case, kind='large', stage='inputs', x=[[0]]*len(stored), fields=stored.tolist(), parameters=dict(depth=0, regularization=regularization))
    retain(tmp_path, 'policy-large-'+case, record)
    with ExecutionContext() as context:
        ops, data, fields = prepare(context, stored)
        def forbidden(*args, **kwargs):
            raise AssertionError('root-only field leaf must not allocate a histogram')
        monkeypatch.setattr(ops, 'histogram', forbidden)
        result = trees.depthwise(ops, data, fields, binning=binning, max_depth=0,
                                field_leaf=lambda o, f, r: leaf(o, f, r, reg_lambda=regularization))
        verify_tree(context, ops, data, result, record, tmp_path, 'large-'+case)


@pytest.mark.parametrize('scale', [-1, 0, 2])
def test_separate_leaf_fields_receive_original_routed_rows(scale, tmp_path):
    with ExecutionContext() as context:
        ops, data, fields, _, binned, x, exact = setup(context)
        values = np.asarray(exact, np.float32)
        values[:, 0] *= scale
        leaves = ops.fields(data, context.upload(values), names=fields.names, roles=fields.roles)
        record = dict(case='captured', kind='separate', scale=scale, stage='inputs', x=x,
                      fields=[[float(v) for v in r] for r in exact], leaf_fields=values.tolist())
        retain(tmp_path, 'policy-separate-'+str(scale), record)
        def explicit_leaf(o, f, r):
            assert f is leaves
            return leaf(o, f, r)
        result = trees.depthwise(ops, data, fields, binning=binned.binning, ordering=exact_order,
                                field_leaf=explicit_leaf, leaf_fields=leaves)
        verify_tree(context, ops, data, result, record, tmp_path, 'separate-'+str(scale))


@pytest.mark.parametrize('policy', ['regularization', 'minimum', 'penalty', 'information'])
def test_callbacks_own_nondefault_numerical_parameters(policy, tmp_path):
    with ExecutionContext() as context:
        ops, data, fields, _, binned, x, exact = setup(context)
        parameters = dict(regularization=dict(regularization=2), minimum=dict(minimum=100),
                          penalty=dict(penalty=100), information=dict(information_minimum=100))[policy]
        options = dict(regularization=dict(reg_lambda=2), minimum=dict(min_child_h=100),
                       penalty=dict(split_penalty=100), information=dict(min_information={'cohort': 100}))[policy]
        saved = [[float(v) for v in r] for r in exact]
        if policy == 'information':
            fields = ops.add_independent(fields, 'cohort', context.upload(np.ones(data.n_rows, np.float32)), nonnegative=True)
            saved = [r+[1.] for r in saved]
        record = dict(case='captured', kind='parameters', policy=policy, stage='inputs', x=x, fields=saved, parameters=parameters)
        retain(tmp_path, 'policy-parameters-'+policy, record)
        result = trees.depthwise(ops, data, fields, binning=binned.binning,
                                ordering=lambda o, b: exact_order(o, b, **options),
                                field_leaf=lambda o, f, r: leaf(o, f, r, reg_lambda=parameters.get('regularization', 1)))
        verify_tree(context, ops, data, result, record, tmp_path, 'parameters-'+policy)


@pytest.mark.parametrize('fault', ['ordering_value', 'field_leaf_value', 'score', 'mask', 'leaf', 'ordering_lambda',
                                  'ordering_minimum', 'ordering_penalty', 'field_lambda'])
def test_exclusive_policy_validation_precedes_growth(fault):
    with ExecutionContext() as context:
        ops, data, fields, _, binned, _, _ = setup(context)
        options = dict(ordering_value=dict(ordering=1), field_leaf_value=dict(field_leaf=1),
                       score=dict(ordering=exact_order, scoring=exact_order), mask=dict(ordering=exact_order, legality=exact_order),
                       leaf=dict(field_leaf=leaf, leaf=leaf), ordering_lambda=dict(ordering=exact_order, reg_lambda=2),
                       ordering_minimum=dict(ordering=exact_order, min_child_h=1), ordering_penalty=dict(ordering=exact_order, split_penalty=1),
                       field_lambda=dict(field_leaf=leaf, reg_lambda=2))[fault]
        before, buffers, records = dict(context.metrics), set(context._buffers), set(ops._records)
        with pytest.raises(ValueError):
            trees.depthwise(ops, data, fields, binning=binned.binning, **options)
        assert context.metrics['kernel_launches'] == before['kernel_launches']
        assert set(context._buffers) == buffers and set(ops._records) == records


@pytest.mark.parametrize('fault', ['forged', 'released', 'wrong_batch', 'released_batch', 'wrong_type', 'empty', 'exception'])
def test_ordering_failures_discard_tree_and_allow_explicit_retry(fault):
    with ExecutionContext() as context:
        ops, data, fields, _, binned, _, _ = setup(context)
        buffers, records, live = set(context._buffers), set(ops._records), context.metrics['live_bytes']
        def broken(o, b):
            context.upload(np.ones(7, np.float32))
            if fault == 'exception':
                raise RuntimeError('injected ordering failure')
            if fault == 'wrong_type':
                return b
            if fault == 'wrong_batch':
                b = o.candidates(b.histogram)
            if fault == 'empty':
                values = np.zeros(b.size, np.float32)
                values[next(i for i in range(b.size) if b.key(i) == (0, 3, True))] = 1
                return o.choose(b, o.scores(b, context.upload(values)), o.mask(b, context.upload(np.ones(b.size, bool))))
            result = exact_order(o, b)
            if fault == 'forged':
                return replace(result)
            if fault == 'released':
                o.release(result)
            elif fault == 'released_batch':
                o.release(b)
            return result
        with pytest.raises((ValueError, RuntimeError)):
            trees.depthwise(ops, data, fields, binning=binned.binning, ordering=broken, field_leaf=leaf)
        assert set(context._buffers) == buffers and set(ops._records) == records and context.metrics['live_bytes'] == live
        result = trees.depthwise(ops, data, fields, binning=binned.binning, ordering=exact_order, field_leaf=leaf)
        assert result.n_nodes == 7


@pytest.mark.parametrize('fault', ['shape', 'dtype', 'nonfinite', 'released', 'forged', 'exception', 'late_exception'])
def test_field_leaf_failure_cleans_completed_nodes_and_allows_explicit_retry(fault):
    with ExecutionContext() as context:
        ops, data, fields, _, binned, _, _ = setup(context)
        buffers, records, live = set(context._buffers), set(ops._records), context.metrics['live_bytes']
        calls = 0
        def broken(o, f, r):
            nonlocal calls
            calls += 1
            if fault == 'late_exception' and calls < 3:
                return leaf(o, f, r)
            context.upload(np.ones(7, np.float32))
            if fault in ('exception', 'late_exception'):
                raise RuntimeError('injected field leaf failure')
            value = context.upload(np.array([np.nan] if fault == 'nonfinite' else [1, 2] if fault == 'shape' else [1],
                                            np.float64 if fault == 'dtype' else np.float32))
            if fault == 'released':
                context.release(value)
            return replace(value) if fault == 'forged' else value
        with pytest.raises((ValueError, RuntimeError)):
            trees.depthwise(ops, data, fields, binning=binned.binning, ordering=exact_order, field_leaf=broken)
        assert set(context._buffers) == buffers and set(ops._records) == records and context.metrics['live_bytes'] == live
        assert trees.depthwise(ops, data, fields, binning=binned.binning, ordering=exact_order, field_leaf=leaf).n_nodes == 7


def test_explicit_no_split_keeps_leaf_and_releases_ordering_scratch():
    with ExecutionContext() as context:
        ops, data, fields, _, binned, _, _ = setup(context)
        buffers, records = set(context._buffers), set(ops._records)
        def no_split(o, b):
            order.rank(o, b)
            return None
        result = trees.depthwise(ops, data, fields, binning=binned.binning, ordering=no_split, field_leaf=leaf)
        assert result.n_nodes == 1 and set(ops._records)-records == {result}
        assert set(context._buffers)-buffers == set(ops._records[result][0])
