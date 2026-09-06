# Numeric preparation and scalar operations

The initial B04 slice exposes CPU binning, named row fields, histograms, candidate
statistics, scoring/feasibility callbacks, routing and scalar Newton leaves.
The [depthwise grower](trees.md) composes these operations into numeric trees.

```python
import numpy as np
from functools import partial
from openboost import NumericData, Problem
from openboost.binning import NumericBinning
from openboost.stats import newton
from openboost.ops import histogram, candidates, choose, feasible, partition, newton_leaf

x = NumericData(np.arange(6)[:, None], np.arange(100, 106), ("x",))
p = Problem(x, np.zeros((6, 1)), x.row_ids)
b = NumericBinning.fit(x, bins=6).transform(x)
fields = newton(p, [-6, 1, 1, 1, 1, 2], np.ones(6))
fields = fields.add_independent("a", [1, 0, 1, 0, 1, 0])
fields = fields.add_independent("b", [0, 1, 0, 1, 0, 1])
options = candidates(histogram(b, fields))
best = choose(options, legality=partial(feasible, min_information={"a": 1, "b": 1}))
left, right = partition(b, None, best)
for rows in (left, right):
    assert fields.values[rows, 2:].sum(axis=0).min() >= 1
    leaf = newton_leaf(histogram(b, fields, rows).total, fields.names)
    assert np.isfinite(leaf)
```

Fit binning on training data, then reuse it on validation/inference data. Cuts use
linear empirical quantiles, duplicate removal and the rule value <= cut goes left.
The minimum observed value may be a cut. Out-of-range observations use end bins.
Constant/all-missing columns have no internal cut; all-missing training columns
produce no candidates. A constant observed column may still split missing rows.
If finite inputs overflow quantile interpolation, fitting fails explicitly rather
than silently dropping those cuts; rescale such extreme features.
Codes are owned feature-major int32 arrays `[F, N]`; missingness is a separate
boolean array. The binning identity distinguishes transformed views of the same
raw data. No weights enter quantile fitting.

`newton(problem, gradient, curvature)` takes unweighted scalar `[N]` derivatives
and applies the original `[N]` training weights once. It does not calculate the
objective derivatives or support vector leaves yet. General `RowFields` declare
names and per-column weight roles. `apply_weight` rejects already weighted fields;
`add_independent` leaves auxiliary fields unweighted. Independent information must
be supplied with the statistical meaning required by the caller's constraint.
Metadata prevents API-level reweighting but cannot prove arbitrary plugin math.

Histograms use selected original positional rows, not rescaled parent summaries.
They preserve physical counts separately from weighted curvature and sum all named
fields. Candidates use histogram prefix/suffix sums and both missing routes.
`score` uses G²/(2(H+lambda)) and one split penalty; `feasible` requires nonempty
children and positive curvature plus declared minima. Information minima require
independent fields. `choose` accepts replacement scoring and legality functions;
it chooses the highest strictly positive gain with exact lexicographic ties by
feature, threshold and missing direction (right first). Invalid custom scores fail.

`partition` returns original positional rows; their source IDs are
`b.data.row_ids[rows]`. Applying a candidate to a different binning or routed row
sequence fails. Leaf solving consumes already-weighted sums without weighting
again. Operations are synchronous CPU NumPy/Python; no CUDA, sparse memory guarantee,
fusion, throughput or complete boosting result is claimed. The public depthwise grower uses
these operations with verified numeric tree persistence.
