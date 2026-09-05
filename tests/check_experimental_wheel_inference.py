"""Generate source fixtures, then verify raw inference with an isolated wheel.

Create: python -m tests.check_experimental_wheel_inference create /tmp/ob-inference
Verify: python -I /absolute/path/to/this_file.py verify /tmp/ob-inference
Run the second command with uv --isolated --with /absolute/path/to/wheel.whl.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

import openboost
from openboost.experimental import Booster, TrainerConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['create', 'verify'])
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    directory = args.directory
    if args.mode == 'create':
        # These training-only fixtures deliberately cannot be pickled.
        from tests.test_experimental_persistence import (
            NoPickleBuilder,
            NoPickleObjective,
            NoPickleSchedule,
        )

        directory.mkdir(parents=True, exist_ok=True)
        X = np.zeros((4, 1), dtype=np.float32)
        model = Booster(objective=NoPickleObjective(), tree_builder=NoPickleBuilder(),
                        step_schedule=NoPickleSchedule(),
                        config=TrainerConfig(n_trees=3, learning_rate=.5)).fit(X, np.ones(4))
        model.save(directory / 'model.ob')
        np.savez(directory / 'expected.npz', X=X, **model.predict_raw(X))
        return
    assert 'site-packages' in openboost.__file__, openboost.__file__
    model = Booster.load(directory / 'model.ob')
    with np.load(directory / 'expected.npz') as data:
        for k, v in model.predict_raw(data['X']).items():
            np.testing.assert_array_equal(v, data[k])
    assert not any(k.startswith('tests.test_experimental') for k in sys.modules)
    assert model.objective is model.tree_builder is model.step_schedule is None
    print('PASS: isolated wheel; plugin-free CPU raw inference; exact two-channel prediction')
    print(openboost.__file__)


if __name__ == '__main__':
    main()
