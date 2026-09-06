# Shared CPU training preparation

PreparedData fits training binning and codes once for immutable feature data.
Every built-in recipe accepts prepared alongside the matching bins setting.
Targets, weights, offsets, objectives and run state remain separate.

```python
from openboost import NumericData, Problem, RunContext
from openboost.binning import PreparedData
from openboost.recipes import squared
from openboost.runs import RunSpec, run_many

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
train = Problem(x, [[0], [2], [1], [6]], x.row_ids)
valid = Problem(v, [[1], [4]], v.row_ids)
prepared = PreparedData(x, bins=4)
jobs = [
    RunSpec(RunContext(f"model-{i}", i), train, valid, squared,
            {"bins": 4, "rounds": i+1}, prepared=prepared)
    for i in range(8)
]
outcomes = run_many(jobs)
assert all(item.error_type is None for item in outcomes)
```

Preparation identity includes training data content/schema/row IDs, bin capacity
and fitted transformer/code identity. Supplied preparation must match both the
recipe's feature data and bins configuration, even if two capacities happen to
produce identical cuts. Mismatches fail rather than refitting silently.
Different targets/weights over identical features may share preparation.

RunSpec has a dedicated prepared field; it cannot be hidden in scalar options.
run_many forwards it to the recipe and records individual failures. Without
preparation, recipes fit their own training binning as before. There is no
implicit cache keyed by object address or task name.

This reuses the training codes consumed by histogram growth. Model inference
still transforms input rows when predicting, and validation preprocessing is
not cached by PreparedData. The record is an in-memory CPU object, not a
serialized training-resume checkpoint or a device workspace.

Tests compare M=1/8/32 heterogeneous jobs with independent, reversed and
regrouped execution, prohibit refitting after preparation, and verify distinct
raw caches, config mismatch rejection and continued execution after a failure.
These are equivalence checks, not timing/fusion evidence. Validation-driven
stopping, real model selection, GPU batching and end-to-end cost remain open.
