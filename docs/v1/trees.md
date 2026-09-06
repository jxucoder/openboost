# Depthwise numeric trees

The CPU grower assembles public histogram, candidate, choice, routing and scalar
leaf operations. It accepts replacement scoring, legality and leaf functions.
Each layer chooses positive-gain splits; when a leaf cap binds, higher gains win,
then node ID and condition break ties. Child IDs are explicit and stable.

```python
import tempfile
from pathlib import Path
import numpy as np
from openboost import NumericData, Problem
from openboost.binning import NumericBinning
from openboost.stats import newton
from openboost.tree import depthwise, NumericTree

x = NumericData([[0], [1], [2], [3], [np.nan]], [10, 11, 12, 13, 14], ("x",))
p = Problem(x, np.zeros((5, 1)), x.row_ids)
b = NumericBinning.fit(x, bins=4).transform(x)
fields = newton(p, [-4, -2, 1, 3, 2], np.ones(5))
tree = depthwise(b, fields, max_depth=2, max_leaves=3)
assert tree.predict(x).shape == (5, 1)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "tree.json"
    tree.save(path)
    restored = NumericTree.load(path)
    np.testing.assert_array_equal(restored.predict(x), tree.predict(x))
```

`scoring(candidate)` and `legality(candidate)` have the public operation contracts.
Scoring executes once per legal candidate, including when ranking nodes within a
layer. `leaf(total, names)` returns one finite scalar from weighted additive sums.
For custom Newton regularization, configure both scoring and leaf solving with
the same regularizer. A custom legality callback must preserve nonempty children;
the grower rejects an admitted empty child. Callbacks should be deterministic.

`NumericTree` owns immutable int32 feature/threshold/child arrays, boolean missing
routes and float64 values. Leaf children, feature and threshold use -1; prediction
uses explicit indices. Construction/load rejects cycles, shared or unreachable
nodes, invalid indices, schema mismatches and nonfinite leaves. Artifacts contain
numeric cuts and ordered feature names, including missing-only split thresholds.
They support unseen numeric values and missing values with the saved transformer.

`predict` returns raw scalar learner output `[N, 1]`. It applies no base, coefficient
or observation offset. These trees are not yet terms in the B03 transaction model;
that integration and complete squared/Normal boosting recipes are next. Artifacts
are for inference, not training resumption. Categories, vector/linear leaves,
best-first/symmetric growth, CUDA and performance claims remain outside this slice.
