"""Historical 90-setting Normal matrix bound to current exact CPU operations."""
from functools import partial

import numpy as np
import pytest

from openboost import RunContext, newton_order
from openboost.recipes import normal
from openboost.tree import depthwise

from .reference.device_normal import fixture
from .reference.exact_growth import fit
from .reference.exact_normal import run as independent_run
from .test_device_normal_reference import prepared_fixture


def configuration(scenario, geometry, update, step):
    case, depth, minimum = dict(weighted_d1=('weighted', 1, None), d2_d1=('d2', 1, None),
                               d2_constrained=('d2', 2, 1), conflict_root=('conflict', 0, None),
                               conflict_d2=('conflict', 2, None))[scenario]
    mode, damping = dict(ordinary=('ordinary', 0), natural=('natural', 0), damped=('natural', .25))[geometry]
    return dict(case=case, depth=depth, minimum=minimum, mode=mode, damping=damping, update=update,
                fixed=step == 'fixed', rate=.1 if step == 'fixed' else 8.)


def inputs(config):
    train, validation, binned, information = prepared_fixture(config['case'])
    f = fixture(config['case'])
    for prefix, p in (('', train), ('validation_', validation)):
        b = binned.binning.transform(p.data)
        f[prefix+'x'] = [[None if b.missing[q, i] else int(b.codes[q, i]) for q in range(len(binned.binning.feature_names))]
                         for i in range(len(p.target))]
    return train, validation, binned, information, f


def flat(result):
    return tuple(sub for outer in result.steps for sub in (outer if isinstance(outer, tuple) else (outer,)))


def tree_keys(tree):
    return [None if f == -1 else (int(f), int(t), bool(m))
            for f, t, m in zip(tree.feature, tree.threshold, tree.missing_left, strict=True)]


def cpu_fit(config):
    train, validation, binned, information, f = inputs(config)
    fitted = []
    minimum = config['minimum']
    constraints = {'cohort:red': 1, 'cohort:blue': 1} if minimum is not None else None
    def learner(prepared, fields):
        # Bind the original historical fitted bins explicitly, with the same
        # immutable training data. Do not forge or mutate PreparedData.
        assert prepared.data.identity == binned.data.identity
        if minimum is not None:
            fields = fields.add_independent('cohort:red', information[:, 0]).add_independent('cohort:blue', information[:, 1])
        tree = depthwise(binned, fields, max_depth=config['depth'],
                         ordering=partial(newton_order.rank, min_information=constraints), field_leaf=newton_order.leaf)
        stored = fit(f['x'], fields.values, policy='depthwise', depth=config['depth'],
                     information_minima={2: 1, 3: 1} if minimum is not None else None)
        assert tree_keys(tree) == [n['key'] for n in stored]
        np.testing.assert_array_equal(tree.left, [n['left'] for n in stored])
        np.testing.assert_array_equal(tree.right, [n['right'] for n in stored])
        np.testing.assert_array_equal(tree.value[:, 0], [float(n['value']) for n in stored])
        fitted.append(dict(fields=fields.values.tolist(), tree=tree.record()))
        return tree
    result = normal(train, validation, context=RunContext('145-current-cpu', 7), rounds=3,
                    learner=learner, mode=config['mode'], damping=config['damping'], update=config['update'],
                    step='fixed' if config['fixed'] else 'backtracking', learning_rate=config['rate'])
    expected = independent_run(f, policy='depthwise', depth=config['depth'], minimum=minimum, mode=config['mode'],
                               damping=config['damping'], update=config['update'], fixed=config['fixed'], rate=config['rate'])
    steps = flat(result)
    assert len(steps) == len(expected['steps'])
    index = 0
    for step, ref in zip(steps, expected['steps'], strict=True):
        assert (step.round_index, step.channels, step.before_version, step.after_version, step.accepted) == (
            ref['round'], ref['channels'], ref['before_version'], ref['version'], ref['accepted'])
        assert step.coefficients == tuple(a[0] for a in ref['attempts'])
        assert [v is not None for v in step.failures] == [a[1] == 'invalid' for a in ref['attempts']]
        for values, name in [(step.raw_before, 'before'), (step.raw_after, 'raw'), (step.gradient, 'gradient'),
                             (step.fisher_diagonal, 'fisher'), (step.direction, 'direction')]:
            np.testing.assert_allclose(values, ref[name], rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose([step.loss_before, step.loss_after], [ref['loss_before'], ref['loss']], rtol=1e-6, atol=1e-8)
        for nodes in ref['nodes']:
            model = fitted[index]['tree']
            keys = [None if a == -1 else (a,b,c) for a,b,c in zip(model['feature'], model['threshold'], model['missing_left'], strict=True)]
            assert keys == [n['key'] for n in nodes]
            np.testing.assert_allclose(np.asarray(model['value'])[:,0], [float(n['value']) for n in nodes], rtol=1e-7, atol=1e-9)
            index += 1
    assert result.stop.completed_rounds == 3 and result.stop.reason == 'budget'
    assert len(result.state.best_model.terms) == expected['best_terms']
    np.testing.assert_allclose(result.state.train_raw, expected['raw'], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result.state.validation_raw, expected['validation'], rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(result.state.best_model.predict(validation.data), expected['best'], rtol=1e-6, atol=1e-8)
    return result, fitted, expected


@pytest.mark.parametrize('scenario', ['weighted_d1', 'd2_d1', 'd2_constrained', 'conflict_root', 'conflict_d2'])
@pytest.mark.parametrize('geometry', ['ordinary', 'natural', 'damped'])
@pytest.mark.parametrize('update', ['joint', 'forward', 'reverse'])
@pytest.mark.parametrize('step', ['fixed', 'backtracking'])
def test_historical_matrix_uses_actual_corrected_cpu_operations(scenario, geometry, update, step):
    cpu_fit(configuration(scenario, geometry, update, step))
