"""A new interpreter after uninstall must infer without training extensions."""

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np

from openboost.experimental import Booster

for name in ("normal_fisher", "bounded_leaves"):
    assert importlib.util.find_spec(name) is None, name
count = 0
for path in sorted(Path("gpu_saved").glob("*.ob")):
    expected = np.load(path.with_suffix(".npz"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = Booster.load(path)
    for channel, actual in model.predict_raw(expected["X"]).items():
        np.testing.assert_array_equal(actual, expected[channel])
    count += 1
assert count == 9, count
print(json.dumps({"extensions_absent": True, "exact_cpu_roundtrips": count}))
