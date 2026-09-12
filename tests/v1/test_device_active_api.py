"""Host configuration contracts; active device phases require real CUDA."""

import subprocess
import sys
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest


def test_public_active_squared_phase_exists():
    from openboost.device_active import SquaredConfiguration, SquaredPhase

    assert callable(SquaredConfiguration) and callable(SquaredPhase)


@pytest.mark.parametrize('step', ['fixed', 'backtracking'])
@pytest.mark.parametrize('rounds', [0, 1, 4])
def test_immutable_independent_configuration(step, rounds):
    from openboost.device_active import SquaredConfiguration

    config = SquaredConfiguration(rounds=rounds, learning_rate=.1, step=step, max_trials=3)
    assert config.rounds == rounds and config.step == step and config.max_trials == 3
    assert config.learning_rate == float(np.float32(.1))
    with pytest.raises(FrozenInstanceError):
        config.rounds = 9
    altered = replace(config, max_depth=0, reg_lambda=2)
    assert altered.max_depth == 0 and config.max_depth == 2


@pytest.mark.parametrize('fault', ['rounds', 'bool_rounds', 'patience', 'bool_patience',
                                  'disabled_delta', 'negative_delta', 'nan_delta', 'depth',
                                  'bool_depth', 'rate', 'nan_rate', 'bool_rate', 'overflow_rate',
                                  'regularization', 'minimum', 'penalty', 'step', 'trials', 'bool_trials'])
def test_invalid_configuration_rejected_without_device_allocation(fault):
    from openboost.device_active import SquaredConfiguration

    options = {'rounds': dict(rounds=-1), 'bool_rounds': dict(rounds=True),
               'patience': dict(patience=0), 'bool_patience': dict(patience=True),
               'disabled_delta': dict(min_delta=1), 'negative_delta': dict(patience=2, min_delta=-1),
               'nan_delta': dict(patience=2, min_delta=np.nan), 'depth': dict(max_depth=-1),
               'bool_depth': dict(max_depth=True), 'rate': dict(learning_rate=-1),
               'nan_rate': dict(learning_rate=np.nan), 'bool_rate': dict(learning_rate=True),
               'overflow_rate': dict(learning_rate=1e100), 'regularization': dict(reg_lambda=-1),
               'minimum': dict(min_child_h=-1), 'penalty': dict(split_penalty=-1),
               'step': dict(step='automatic'), 'trials': dict(max_trials=7), 'bool_trials': dict(max_trials=True)}[fault]
    with pytest.raises(ValueError):
        SquaredConfiguration(**options)


def test_import_does_not_require_cuda_packages():
    code = """
import sys
for name in ('cupy', 'numba'):
    sys.modules[name] = None
from openboost.device_active import SquaredConfiguration, SquaredPhase
assert SquaredConfiguration().rounds == 2
assert all(callable(getattr(SquaredPhase, n)) for n in ('request_tree', 'advance', 'result', 'close'))
"""
    subprocess.run([sys.executable, '-I', '-c', code], check=True, capture_output=True)


def test_wrong_configuration_type_rejects_before_run_construction():
    from openboost.device_active import SquaredPhase

    with pytest.raises(ValueError, match='SquaredConfiguration'):
        SquaredPhase(None, None, None, run_id='never-started', seed=0, configuration={})
