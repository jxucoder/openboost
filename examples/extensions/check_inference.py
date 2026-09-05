"""Run in a fresh interpreter after uninstalling both training extensions."""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np

from openboost.experimental import Booster

for name in ("normal_fisher", "bounded_leaves"):
    assert importlib.util.find_spec(name) is None, name
count = 0
for model_path in sorted(Path("saved").glob("*.ob")) + [Path("demo.ob")]:
    expected = np.load(model_path.with_suffix(".npz"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = Booster.load(model_path)
    for channel, actual in model.predict_raw(expected["X"]).items():
        np.testing.assert_array_equal(actual, expected[channel])
    count += 1
assert count == 6
print(json.dumps({"extensions_absent": True, "exact_cpu_roundtrips": count}))
