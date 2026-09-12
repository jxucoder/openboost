"""Exact original-row CUDA order, public routing/tree consumer and failure ownership."""

import json
import os
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pytest

from openboost import NumericData, Problem
from openboost import device_newton_order as exact_order
from openboost import device_tree as trees
from openboost.binning import Binning
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.normal_precision import ATOL, RTOL
from .reference.normal_split_order import candidates, exact_tree, predict, winner
from .test_normal_order_current_reference import prepared, topology

pytestmark = pytest.mark.gpu


def integer(digits):
    return sum(int(d) << (16*i) for i, d in enumerate(digits))


def retain(tmp_path, name, record):
    root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'exact-newton'
    root.mkdir(parents=True, exist_ok=True)
    (root / (name+'.json')).write_text(json.dumps(record, indent=2)+'\n')


def setup(context, case='captured'):
    problem, binned, host_fields, x, exact = prepared(case)
    ops = DeviceOperations(context)
    data = ops.prepare(binned, problem)
    buffer = context.upload(host_fields.values.astype(np.float32))
    fields = ops.fields(data, buffer, names=host_fields.names, roles=host_fields.roles)
    return ops, data, fields, problem, binned, x, exact


def exact_tree_consumer(ops, data, fields, binned):
    """An ordinary public loop; existing grower/default bodies remain unchanged."""
    queue, nodes, values, retained = [(ops.rows(data), 0)], [], [], []
    for rows, level in queue:
        histogram = ops.histogram(data, fields, rows)
        value = ops.leaf(histogram)
        values.append(value)
        node = [-1, -1, False, -1, -1]
        if level < 2:
            batch = ops.candidates(histogram)
            ordering = exact_order.rank(ops, batch)
            split = exact_order.choose(ops, ordering)
            if split is not None:
                left, right = ops.partition(rows, split)
                node = [*split.key, len(queue), len(queue)+1]
                queue.extend(((left, level+1), (right, level+1)))
                ops.release(split)
            ops.release(ordering)
            ops.release(batch)
        nodes.append(tuple(node))
        retained.extend((histogram, rows))
    result = trees.assemble(ops, binning=binned.binning, topology=tuple(nodes), values=tuple(values))
    for value in values:
        ops.execution.release(value)
    for record in retained:
        ops.release(record)
    return result


@pytest.mark.parametrize('case', ['captured', 'p24-negative', 'p24-zero', 'p24-positive',
                                  'p54-negative', 'p54-zero', 'p54-positive',
                                  'p100-negative', 'p100-zero', 'p100-positive',
                                  'p149-negative', 'p149-zero', 'p149-positive'])
def test_exact_root_routing_and_public_tree_consumer(case, tmp_path):
    with ExecutionContext() as context:
        ops, data, fields, problem, binned, x, exact = setup(context, case)
        record = dict(case=case, stage='inputs', x=x, fields=[[float(v) for v in row] for row in exact],
                      row_ids=problem.data.row_ids.tolist())
        retain(tmp_path, case, record)
        rows = ops.rows(data)
        histogram = ops.histogram(data, fields, rows)
        batch = ops.candidates(histogram)
        metrics = dict(context.metrics)
        ordering = exact_order.rank(ops, batch)
        selected = exact_order.choose(ops, ordering)
        decision_metrics = dict(context.metrics)
        sums, p, q, eligible = [context.export(handle) for handle in
                               (ordering.sums, ordering.numerator, ordering.denominator, ordering.eligible)]
        parts = None if selected is None else ops.partition(rows, selected)
        record.update(stage='root', selected_key=None if selected is None else selected.key,
                      sums=sums.tolist(), numerator=p.tolist(), denominator=q.tolist(), eligible=eligible.tolist(),
                      partitions=None if parts is None else [context.export(side.positions).tolist() for side in parts],
                      before_metrics=metrics, decision_metrics=decision_metrics)
        retain(tmp_path, case, record)
        tree = exact_tree_consumer(ops, data, fields, binned)
        exported = trees.export(ops, tree)
        prediction = context.export(trees.predict(ops, tree, data))
        record.update(stage='complete', tree=exported.record(), prediction=prediction.tolist())
        retain(tmp_path, case, record)
        options = candidates(x, exact, range(8))
        wanted = winner(options)
        assert selected is not None and selected.key == wanted['key']
        assert record['partitions'] == [list(side) for side in wanted['rows']]
        # Every scalar aggregate and rational child-score ratio is independent
        # original-row arithmetic, rather than a replay of kernel operations.
        for i, expected in enumerate(options):
            assert batch.key(i) == expected['key']
            for side, rows in enumerate(expected['rows']):
                wanted_sums = [sum((max(exact[r][0], F(0)) for r in rows), F(0)),
                               sum((max(-exact[r][0], F(0)) for r in rows), F(0)),
                               sum((exact[r][1] for r in rows), F(0))]
                assert [F(integer(sums[i, side*3+j]), 2**149) for j in range(3)] == wanted_sums
            assert bool(eligible[i]) == (expected['legal'] and expected['gain'] > 0)
            if expected['legal']:
                assert F(integer(p[i]), 2*2**149*integer(q[i])) == sum(g*g/(2*(h+1)) for g, h in expected['sums'])
        wanted_tree = exact_tree(x, exact, depth=2)
        assert list(tree.topology) == topology(wanted_tree)
        np.testing.assert_allclose(exported.value[:, 0], [float(n['value']) for n in wanted_tree], rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(prediction[:, 0], [float(v) for v in predict(wanted_tree, x)], rtol=RTOL, atol=ATOL)
        exported_bytes = decision_metrics['export_bytes']-metrics['export_bytes']
        validation_bytes = decision_metrics['validation_export_bytes']-metrics['validation_export_bytes']
        assert exported_bytes == validation_bytes+4
        assert decision_metrics['decision_export_bytes']-metrics['decision_export_bytes'] == 4


@pytest.mark.parametrize('fault', ['foreign_order', 'released_order', 'released_fields', 'released_rows', 'foreign_mask'])
def test_invalid_owner_or_binding_rejects_without_new_work(fault):
    with ExecutionContext() as context:
        ops, data, fields, _, _, _, _ = setup(context)
        rows = ops.rows(data)
        batch = ops.candidates(ops.histogram(data, fields, rows))
        ordering = exact_order.rank(ops, batch)
        mask = None
        if fault == 'foreign_order':
            ordering = replace(ordering)
        elif fault == 'released_order':
            ops.release(ordering)
        elif fault == 'released_fields':
            ops.release(fields)
        elif fault == 'released_rows':
            ops.release(rows)
        else:
            other = ops.candidates(ops.histogram(data, fields, ops.rows(data)))
            mask = ops.feasible(other)
        handles, records, metrics = set(context._buffers), set(ops._records), dict(context.metrics)
        with pytest.raises(ValueError):
            exact_order.choose(ops, ordering, mask)
        assert set(context._buffers) == handles and set(ops._records) == records
        assert context.metrics['live_bytes'] == metrics['live_bytes']
        assert context.metrics['kernel_launches'] == metrics['kernel_launches']


@pytest.mark.parametrize('index', [1, 3, 5, 7])
def test_partial_allocation_failure_preserves_caller_then_same_batch_retry(index, monkeypatch):
    with ExecutionContext() as context:
        ops, data, fields, _, _, x, exact = setup(context, 'p149-positive')
        batch = ops.candidates(ops.histogram(data, fields, ops.rows(data)))
        handles, records, before = set(context._buffers), set(ops._records), context.metrics['live_bytes']
        original, calls = context._empty, 0
        with monkeypatch.context() as patch:
            def fail(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == index:
                    raise MemoryError('injected exact order allocation failure')
                return original(*args, **kwargs)
            patch.setattr(context, '_empty', fail)
            with pytest.raises(MemoryError, match='injected exact'):
                exact_order.rank(ops, batch)
        assert set(context._buffers) == handles and set(ops._records) == records
        assert context.metrics['live_bytes'] == before
        ordering = exact_order.rank(ops, batch)
        assert exact_order.choose(ops, ordering).key == winner(candidates(x, exact, range(8)))['key']


@pytest.mark.parametrize('policy', ['minimum', 'penalty', 'information', 'all_masked'])
def test_exact_constraints_and_extra_mask_can_remove_every_split(policy):
    with ExecutionContext() as context:
        ops, data, fields, _, _, _, _ = setup(context)
        options = {}
        if policy == 'information':
            info = context.upload(np.zeros(data.n_rows, np.float32))
            fields = ops.add_independent(fields, 'cohort', info, nonnegative=True)
            options = dict(min_information={'cohort': 1})
        elif policy == 'minimum':
            options = dict(min_child_h=100)
        elif policy == 'penalty':
            options = dict(split_penalty=100)
        batch = ops.candidates(ops.histogram(data, fields, ops.rows(data)))
        ordering = exact_order.rank(ops, batch, **options)
        mask = ops.mask(batch, context.upload(np.zeros(batch.size, bool))) if policy == 'all_masked' else None
        assert exact_order.choose(ops, ordering, mask) is None


@pytest.mark.parametrize('field', ['curvature', 'information'])
def test_mass_rounded_up_to_minimum_is_still_exactly_infeasible(field, tmp_path):
    data = NumericData([[0], [0], [1], [1]], [10, 20, 30, 40], ('x',))
    almost = np.nextafter(np.float32(.5), np.float32(0))
    curvature = np.array([.5, almost, 1, 1], np.float32) if field == 'curvature' else np.ones(4, np.float32)
    stored = np.column_stack(([-1, -1, 1, 1], curvature, [.5, almost, 1, 1])).astype(np.float32)
    problem = Problem(data, [[0]]*4, data.row_ids, weight=curvature)
    binned = Binning(('x',), (np.array([.5]),)).transform(data)
    options = dict(min_child_h=1) if field == 'curvature' else dict(min_information={'cohort': 1})
    payload = dict(stage='inputs', field=field, x=data.values.tolist(), row_ids=data.row_ids.tolist(),
                   fields=stored.tolist(), options=options)
    retain(tmp_path, 'minimum-'+field, payload)
    with ExecutionContext() as context:
        ops = DeviceOperations(context)
        prepared_data = ops.prepare(binned, problem)
        fields = ops.fields(prepared_data, context.upload(stored), names=('gradient', 'curvature', 'cohort'),
                            roles=('training', 'training', 'independent'))
        batch = ops.candidates(ops.histogram(prepared_data, fields, ops.rows(prepared_data)))
        unconstrained = exact_order.rank(ops, batch)
        available = exact_order.choose(ops, unconstrained)
        constrained = exact_order.rank(ops, batch, **options)
        selected = exact_order.choose(ops, constrained)
        payload.update(stage='complete', available=None if available is None else available.key,
                       selected=None if selected is None else selected.key,
                       sums=context.export(constrained.sums).tolist(),
                       eligible=context.export(constrained.eligible).tolist())
        retain(tmp_path, 'minimum-'+field, payload)
        assert F(float(stored[0, 1 if field == 'curvature' else 2])) + F(float(almost)) < 1
        assert available is not None and selected is None
