"""Fresh-process inference after both extension distributions are uninstalled."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from openboost import MixedData
from openboost.artifacts import Model

for name in ("ob_cohort_splits", "ob_penalized_leaves"):
    assert importlib.util.find_spec(name) is None
root = Path(sys.argv[1])
record = json.loads((root / "checks.json").read_text())
data = MixedData(
    record["values"], record["row_ids"], tuple(record["names"]), tuple(record["kinds"])
)
for name, expected in record["predictions"].items():
    np.testing.assert_array_equal(Model.load(root / f"{name}.json").predict(data), expected)
print("Both models preserve exact predictions without extension imports.")
