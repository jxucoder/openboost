# Tree growth policies

The CPU growers assemble public histogram, candidate, choice, routing and scalar/vector
leaf operations. Each accepts replacement scoring, legality and leaf functions.
Child IDs are explicit and stable.

- `depthwise`: choose positive-gain splits within each layer; a leaf cap selects
  higher gains, with node ID and condition breaking ties.
- `best_first`: a heap selects the highest-gain active leaf across depths, then
  node ID and condition. Only newly created children need candidate evaluation.
- `symmetric`: choose one common condition legal in every active leaf, maximizing
  total gain across the complete layer. Individual gains may be negative; their
  total must be positive. A leaf budget must admit the entire next layer.

Scoring and legality callbacks should be pure functions of candidate statistics
and immutable configuration. Unchanged best-first candidates retain cached scores;
changing behavior by call order is unsupported. All three policies preserve
exact condition tie rules and use the same inference format.

```python
import tempfile
from pathlib import Path
import numpy as np
from openboost import NumericData, Problem
from openboost.binning import Binning
from openboost.stats import newton
from openboost.tree import depthwise, Tree

x = NumericData([[0], [1], [2], [3], [np.nan]], [10, 11, 12, 13, 14], ("x",))
p = Problem(x, np.zeros((5, 1)), x.row_ids)
b = Binning.fit(x, bins=4).transform(x)
fields = newton(p, [-4, -2, 1, 3, 2], np.ones(5))
tree = depthwise(b, fields, max_depth=2, max_leaves=3)
assert tree.predict(x).shape == (5, 1)
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "tree.json"
    tree.save(path)
    restored = Tree.load(path)
    np.testing.assert_array_equal(restored.predict(x), tree.predict(x))
```

`scoring(candidate)` and `legality(candidate)` have the public operation contracts.
Scoring executes once per legal candidate, including when ranking nodes within a
layer. `leaf(total, names)` returns a finite scalar or nonempty vector from weighted
additive sums. Every node must return the same width. Optional `leaf_fields` from
the same problem separates split statistics from full leaf statistics. Vector
Newton adapters and callbacks are described in [multiclass and vectors](multiclass.md).
For custom Newton regularization, configure both scoring and leaf solving with
the same regularizer. A custom legality callback must preserve nonempty children;
the grower rejects an admitted empty child. Callbacks should be deterministic.

`Tree` owns immutable int32 feature/threshold/child arrays, boolean missing
routes and float64 values shaped [nodes, L]. Leaf children, feature and threshold use -1; prediction
uses explicit indices. Construction/load rejects cycles, shared or unreachable
nodes, invalid indices, schema mismatches and nonfinite leaves. Artifacts contain
numeric cuts, typed category dictionaries and ordered feature names. Numeric
conditions use <=, categorical conditions use equality. The saved transformer
defines condition kinds and routes unknown tokens as missing.

`predict` returns raw learner output `[N, L]` (L=1 for scalar leaves). It applies no base, coefficient
or observation offset. Mapped tree terms integrate with the transaction model and the
[squared](squared.md) and [Normal](normal.md) recipes. Artifacts
are for inference, not training resumption. Linear leaves,
CUDA and performance claims remain outside this slice.


Recipes accept a custom learner, so growth policy changes do not require editing
the objective or transaction loop:

```python
from functools import partial
from openboost import NumericData, Problem, RunContext
from openboost.recipes import squared
from openboost.tree import best_first, symmetric

x = NumericData([[0, 0], [0, 1], [1, 0], [1, 1]], [1, 2, 3, 4], ("a", "b"))
p = Problem(x, [[-3], [-1], [1], [3]], x.row_ids)
for grow in (best_first, symmetric):
    result = squared(p, p, context=RunContext(grow.__name__, 1), rounds=2,
                     learner=partial(grow, max_depth=3, max_leaves=4))
    assert result.state.version == 2
```

This arithmetic example reuses train/validation data; real evaluation needs the
prescribed distinct partitions. These growth policies do not establish LightGBM
or CatBoost feature/quality parity. [Categorical support](categorical.md) uses the same three policies.
