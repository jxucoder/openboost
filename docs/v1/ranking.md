# Query-local ranking

Ranking consumes scalar nonnegative integer relevance and explicit query roles.
Query IDs are aligned integer codes stored in structure, never appended to features.
Optional query_weight is repeated on each row and must be constant within a query.
Non-unit row weights are rejected. Default query weights are one.

```python
import numpy as np
from openboost import NumericData, Problem, RunContext
from openboost.recipes import ranking

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
train = Problem(x, [[0], [2], [1], [0]], x.row_ids,
                structure={"query": [[0], [0], [1], [1]]})
valid = Problem(v, [[0], [2]], v.row_ids,
                structure={"query": [[2], [2]]})
fit = ranking(train, valid, context=RunContext("ranking-demo", 7),
              rounds=3, lambdas=True, k=10)
scores = fit.state.best_model.predict(v)
assert scores.shape == (2, 1) and np.isfinite(scores).all()
```

Ranking(lambdas=False).geometry(problem, raw) returns pairwise logistic loss and
row gradient/diagonal-curvature vectors. For each query, it enumerates every pair
with unequal relevance, orients the higher-relevance row first, and divides each
pair's weight by that query's eligible pair count. The query weight multiplies
this factor; tied relevance contributes no pairs. Each pair adds opposite gradients
and equal diagonal curvature to its two rows.

With lambdas=True, the factor also includes the absolute change in NDCG@k from
swapping the pair. Gains are 2**relevance - 1; ranks sort by descending score, with
row ID breaking ties. The geometry call freezes these delta weights rather than
differentiating them. Each boosting round recomputes them from the current scores.
Offsets enter scores exactly once. Query-local geometry reduces to ordinary
scalar Newton fields and uses the existing growth and transaction operations.

The recipe uses finite fixed steps. Validation selects best_model using one minus
query-weighted mean NDCG@k; it does not select by the moving lambda-weighted pair
loss. Zero-ideal-DCG queries have NDCG one. All-zero query weights are rejected.
Backtracking, explicit pair weights and pair sampling are not accepted by this
initial API. Enumeration needs quadratic memory and time per query, so it is a
correctness implementation, not a scalable ranking performance claim.

Inference artifacts return raw scores without training query roles. Supply offsets
explicitly when needed; downstream ranking should use the same row-ID tie rule.
Keep complete queries in separate train/validation/test partitions for real
evaluation. The low-level recipe does not certify dataset partition independence.

Independent tests verify pair geometry, logistic finite differences, query
isolation, rank tie permutations, three-round tree composition, offsets and
fresh-process persistence. Real A4 evaluation, quantile/penalized leaves, CUDA
and external-library quality/performance comparisons remain open.
