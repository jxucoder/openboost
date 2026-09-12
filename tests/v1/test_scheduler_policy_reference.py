"""Host prerequisite for the corrected consumed mixed-policy fixture."""

from dataclasses import replace

import pytest

from openboost.device_group_runs import schedule_plan

from .test_grouped_runs_api import prepared


def mixed(specs):
    return tuple(replace(s, options=dict(s.options, max_depth=i%3, learning_rate=(8, 64, .5)[i%3],
                                        step='backtracking', max_trials=6, rounds=2, patience=None, min_delta=0,
                                        reg_lambda=(1, 2)[i%2], split_penalty=.25)) for i, s in enumerate(specs))


@pytest.mark.parametrize('index', [0, 1, 2, 3, 4, 5])
def test_each_corrected_mixed_policy_validates_before_device_work(index):
    specs = mixed(prepared(6))
    config = schedule_plan(specs).configurations[index]
    assert config.patience is None and config.min_delta == 0
    assert config.rounds == 2 and config.max_depth == index%3
    assert config.learning_rate == (8, 64, .5)[index%3]
    assert config.step == 'backtracking' and config.max_trials == 6
    assert config.reg_lambda == (1, 2)[index%2] and config.split_penalty == .25


def test_original_inherited_delta_with_disabled_patience_rejects_on_host():
    specs = mixed(prepared(6))
    bad = replace(specs[1], options=dict(specs[1].options, min_delta=100))
    with pytest.raises(ValueError, match='requires enabled patience'):
        schedule_plan((specs[0], bad))
