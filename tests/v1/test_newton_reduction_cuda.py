"""Parallel exact choice, odd tails, capacity failures and public scratch ownership."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from openboost import device_newton_order as exact_order
from openboost.device import DeviceOperations
from openboost.execution import ExecutionContext

from .reference.newton_reduction import example, upload_words, winner
from .test_device_newton_order_cuda import setup

pytestmark = pytest.mark.gpu


def retain(tmp_path, name, record):
    root = Path(os.environ.get('OPENBOOST_NORMAL_ARTIFACTS', tmp_path)) / 'newton-reduction'
    root.mkdir(parents=True, exist_ok=True)
    (root / (name + '.json')).write_text(json.dumps(record, indent=2) + '\n')


def reduce(ops, pairs, eligible, extra):
    context = ops.execution
    p = context.upload(upload_words([a for a, _ in pairs]))
    q = context.upload(upload_words([b for _, b in pairs]))
    # The public upload boundary supports int32, not uint32. These owned views
    # preserve every limb bit and exercise the production uint32 specialization.
    numerator = context._array(p).view(np.uint32)
    denominator = context._array(q).view(np.uint32)
    e = context.upload(np.asarray(eligible, bool))
    m = context.upload(np.asarray(extra, bool))
    size = max(1, (len(pairs) + 1) // 2)
    output = context._empty((size,), np.int32)
    ops._launch('exact_newton_choose', size, numerator, denominator,
                context._array(e), context._array(m), context._array(output))
    levels = [context.export(output).tolist()]
    while size > 1:
        size = (size + 1) // 2
        previous, output = output, context._empty((size,), np.int32)
        ops._launch('exact_newton_reduce', size, numerator, denominator,
                    context._array(previous), context._array(output))
        levels.append(context.export(output).tolist())
    return levels


@pytest.mark.parametrize('case', ['single', 'three', 'five', 'seven', 'odd_levels',
                                'block_tail', 'housing_width', 'ties', 'masked_ties',
                                'all_ineligible', 'all_masked', 'one_eligible',
                                'masked_maximum', 'sub_ulp_gap', 'large_cross_product'])
def test_parallel_rational_winner_and_every_odd_tail(case, tmp_path):
    pairs, eligible, extra = example(case)
    record = dict(case=case, stage='inputs', pairs=[[hex(a), hex(b)] for a, b in pairs],
                  eligible=eligible, extra=extra)
    retain(tmp_path, case, record)
    with ExecutionContext() as context:
        levels = reduce(DeviceOperations(context), pairs, eligible, extra)
    record.update(stage='complete', levels=levels)
    retain(tmp_path, case, record)
    assert levels[-1] == [winner(pairs, eligible, extra)]
    # Independently enumerate each contiguous original range, including carried
    # odd tails. This detects mistakes hidden by an otherwise correct final max.
    for depth, level in enumerate(levels, 1):
        width = 2**depth
        expected = []
        for start in range(0, len(pairs), width):
            local = winner(pairs[start:start+width], eligible[start:start+width], extra[start:start+width])
            expected.append(-1 if local == -1 else start + local)
        assert level == expected


@pytest.mark.parametrize('fault', ['overflow_first', 'overflow_later', 'zero_denominator', 'masked_invalid'])
def test_capacity_rejection_propagates_across_levels(fault, tmp_path):
    pairs, eligible, extra = [(1, 1)] * 9, [True] * 9, [True] * 9
    if fault == 'overflow_later':
        # Each first-level comparison fits; a subsequent cross product does not.
        pairs[:4] = [(2**1599, 1)] * 2 + [(1, 2**1599)] * 2
    elif fault == 'zero_denominator':
        pairs[6] = (1, 0)
    else:
        pairs[0:2] = [(2**1599, 2**1599)] * 2
        if fault == 'masked_invalid':
            extra[:2] = [False, False]
    record = dict(case=fault, pairs=[[hex(a), hex(b)] for a, b in pairs], eligible=eligible, extra=extra)
    with ExecutionContext() as context:
        levels = reduce(DeviceOperations(context), pairs, eligible, extra)
    record.update(stage='complete', levels=levels)
    retain(tmp_path, fault, record)
    assert levels[-1] == [2 if fault == 'masked_invalid' else -2]


@pytest.mark.parametrize('fault', ['none', 'allocation_first', 'allocation_later', 'launch_later', 'capacity'])
def test_public_choice_releases_all_scratch_and_preserves_borrowed_order(fault, monkeypatch, tmp_path):
    with ExecutionContext() as context:
        ops, data, fields, _, _, _, _ = setup(context, 'p149-positive')
        batch = ops.candidates(ops.histogram(data, fields, ops.rows(data)))
        ordering = exact_order.rank(ops, batch)
        buffers, records = set(context._buffers), set(ops._records)
        before = context.metrics['live_bytes']
        with monkeypatch.context() as patch:
            if fault.startswith('allocation'):
                original, calls = context._empty, 0
                def fail(*args, **kwargs):
                    nonlocal calls
                    calls += 1
                    if calls == (1 if fault == 'allocation_first' else 3):
                        raise MemoryError('injected chooser allocation')
                    return original(*args, **kwargs)
                patch.setattr(context, '_empty', fail)
            elif fault == 'launch_later':
                original = ops._launch
                def fail(name, *args):
                    if name == 'exact_newton_reduce':
                        raise RuntimeError('injected chooser launch')
                    return original(name, *args)
                patch.setattr(ops, '_launch', fail)
            elif fault == 'capacity':
                original = ops._compact
                def fail(handle):
                    original(handle)
                    return np.array([-2], np.int32)
                patch.setattr(ops, '_compact', fail)
            if fault == 'none':
                split = exact_order.choose(ops, ordering)
                assert split.candidates is batch and split.key == (0, 3, False)
                ops.release(split)
            else:
                error = MemoryError if fault.startswith('allocation') else RuntimeError if fault == 'launch_later' else ValueError
                with pytest.raises(error):
                    exact_order.choose(ops, ordering)
        assert set(context._buffers) == buffers and set(ops._records) == records
        assert context.metrics['live_bytes'] == before
        split = exact_order.choose(ops, ordering)
        assert split.candidates is batch and split.key == (0, 3, False)
        ops.release(split)
        assert set(context._buffers) == buffers and set(ops._records) == records
        retain(tmp_path, 'ownership-' + fault, dict(stage='complete', fault=fault, selected_key=[0, 3, False],
               before_bytes=before, after_bytes=context.metrics['live_bytes'], caller_records_preserved=True))
