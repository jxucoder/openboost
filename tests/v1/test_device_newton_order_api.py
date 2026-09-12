"""Exact resident Newton configuration checks; no CUDA emulation."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest


def test_public_exact_operation_and_independent_immutable_parameters():
    from openboost.device_newton_order import NewtonParameters, choose, rank

    source = {'cohort': .1}
    p = NewtonParameters(min_information=source)
    source['cohort'] = 99
    assert p.min_information == (('cohort', float(np.float32(.1))),)
    assert callable(rank) and callable(choose)
    with pytest.raises(FrozenInstanceError):
        p.reg_lambda = 9


@pytest.mark.parametrize('fault', ['negative_lambda', 'infinite_lambda', 'boolean_lambda', 'negative_minimum',
                                  'nan_penalty', 'information_type', 'information_name', 'information_negative',
                                  'information_boolean', 'information_overflow'])
def test_invalid_policy_fails_before_device_allocation(fault):
    from openboost.device_newton_order import NewtonParameters

    option = dict(negative_lambda=dict(reg_lambda=-1), infinite_lambda=dict(reg_lambda=np.inf),
                  boolean_lambda=dict(reg_lambda=True), negative_minimum=dict(min_child_h=-1),
                  nan_penalty=dict(split_penalty=np.nan), information_type=dict(min_information=[]),
                  information_name=dict(min_information={1: 0}), information_negative=dict(min_information={'x': -1}),
                  information_boolean=dict(min_information={'x': True}), information_overflow=dict(min_information={'x': 1e100}))[fault]
    with pytest.raises(ValueError):
        NewtonParameters(**option)


def test_schema_keeps_objective_and_independent_information_roles_distinct():
    from openboost.device_newton_order import NewtonParameters, columns

    p = NewtonParameters(min_information={'cohort': 2})
    assert columns(('cohort', 'curvature', 'gradient'), ('independent', 'training', 'training'), p) == (2, 1, (0,))
    for names, roles in [(('gradient', 'curvature'), ('training', 'training')),
                         (('cohort', 'curvature', 'gradient'), ('training', 'training', 'training')),
                         (('cohort', 'curvature', 'gradient'), ('independent', 'training', 'unweighted'))]:
        with pytest.raises(ValueError):
            columns(names, roles, p)
