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
lexicographic ties of the supplied scores and use the same inference format.
Ordinary histogram scores may lose a mathematical tie or small gain difference;
use the explicit exact ordering operation when that distinction is required.

`ops.newton_choice(histogram, reg_lambda=1, min_child_h=0, split_penalty=0)`
returns an owned `(Candidate, gain)` pair, or `None`. It composes the ordinary
Newton rules directly over numeric histogram arrays, preserving the existing
prefix/suffix reductions, half-square arithmetic order and exact floating-point
tie comparisons. Categorical features and optional `min_information` constraints
use the public exhaustive operations. It does not use exact-rational ranking.

`depthwise(..., selection=callback)` supplies the actual node histogram to a
callback returning that pair or `None`. The grower checks the condition, original
row/data identity, field names/roles, exact histogram statistics and nonempty
child counts, then owns canonical candidate buffers. Gains must be finite and
strictly positive. The callback replaces `scoring`, `legality` and `ordering`,
which must retain their defaults; leaf solving remains separately configurable.
The callback is cooperative: binding checks do not certify an arbitrary supplied
gain or prove that a custom selector searched every condition. Keep its scoring
configuration fixed across nodes. The standard scalar CPU recipe configuration
explicitly composes `newton_choice` through this hook with matching leaf
regularization. Normal retains its exact ordering and original-row leaf policy;
custom callbacks and the other growth policies remain available.

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
the grower rejects an admitted empty child. Callbacks should be deterministic. For residual-based solvers, use the paired
row_leaf and leaf_context arguments described in [quantile leaves](quantile.md).

For exact scalar Newton decisions, supply `ordering(data, fields, rows)` returning
records from [newton_order.rank](newton-order.md). Growth passes the actual node's
original rows and split fields. The hook replaces scoring and feasibility, so
`scoring` and `legality` must retain their defaults. A callback may filter or reorder
records from that evaluation. It runs once per evaluated node; root-only trees
need no candidate evaluation. Empty results leave a node unsplit.

```python
from functools import partial
from openboost.newton_order import rank
from openboost.ops import newton_leaf
from openboost.tree import best_first

exact_tree = best_first(
    b, fields, max_depth=3, max_leaves=4,
    ordering=partial(rank, reg_lambda=2, min_child_h=0),
    leaf=partial(newton_leaf, reg_lambda=2),
)
assert exact_tree.predict(x).shape == (5, 1)
```

Depthwise leaf budgets and the best-first heap compare Fraction gains directly.
Symmetric growth sums all common-condition gains exactly, including negative
per-node values, and requires a strictly positive layer sum. There is no tolerance
band or conversion to float, even when a gain exceeds binary64's finite range.
The operation rejects foreign node/field inputs, duplicate or mixed evaluations,
and configuration changes across nodes. Records are cooperative public values;
these checks do not certify arbitrary fabricated scores. Score settings must be
fixed for gains from different nodes to remain comparable.

Leaf fields, vector outputs and routed leaf solvers remain separate operations.
`field_leaf(fields, rows)` receives the node's immutable RowFields and original row
positions, and returns a finite scalar or vector. It replaces additive `leaf` and
residual-context `row_leaf`, so those callbacks must retain their defaults. With
`leaf_fields`, the callback receives that separate field set, validated against
the same data/problem identity. It does not require a floating histogram reduction.

Use `field_leaf=openboost.newton_order.leaf` for an exact original-row Newton
solution; configure its regularizer alongside the ordering regularizer. Exact
ordering alone leaves the additive leaf operation unchanged. Normal's default
CPU learner now uses both exact operations. Device execution and cost remain
unqualified by these CPU checks. Existing custom scoring and
legality callbacks retain their original behavior when `ordering` is omitted.

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
