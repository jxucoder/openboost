"""Fresh-process inference after all extension distributions are uninstalled."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from openboost import MixedData, NumericData
from openboost.artifacts import Model

for name in ("ob_cohort_splits", "ob_penalized_leaves", "ob_ordered_updates", "ob_expectile"):
    assert importlib.util.find_spec(name) is None
root = Path(sys.argv[1])
record = json.loads((root / "checks.json").read_text())
data = MixedData(
    record["values"], record["row_ids"], tuple(record["names"]), tuple(record["kinds"])
)
for name, expected in record["predictions"].items():
    np.testing.assert_array_equal(Model.load(root / f"{name}.json").predict(data), expected)
ordered = json.loads((root / "ordered-checks.json").read_text())
numeric = NumericData(ordered["values"], np.arange(6), ("feature",))
for name, expected in ordered["predictions"].items():
    np.testing.assert_array_equal(Model.load(root / f"{name}.json").predict(numeric), expected)
expectile = json.loads((root / "expectile-checks.json").read_text())
numeric = NumericData(expectile["values"], np.arange(6), ("feature",))
np.testing.assert_array_equal(
    Model.load(root / "expectile-model.json").predict(numeric), expectile["raw"]
)
assert not Path(__file__).with_name("custom_stopping.py").exists()
threshold = json.loads((root / "threshold-inference.json").read_text())
numeric = NumericData(threshold["values"], threshold["row_ids"], tuple(threshold["names"]))
np.testing.assert_array_equal(
    Model.load(root / "threshold-model.json").predict(numeric), threshold["predictions"]
)
print("Ten models preserve exact predictions without training extensions or custom policy source.")
