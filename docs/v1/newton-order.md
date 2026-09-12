# Exact CPU Newton ordering

`openboost.newton_order` ranks scalar Newton splits using exact sums of the
stored row fields. It resolves mathematical ties lexicographically and preserves
genuine gain differences that disappear when scores are rounded to float.
Use it explicitly through the CPU growth `ordering` hook. Grower defaults retain
floating operations; the Normal CPU recipe uses exact ordering and exact original-row
leaves by default. The separate [device ordering](exact-device-newton.md) and
[device leaf](exact-device-leaf.md) interfaces serve resident callers. End-to-end
cost and exact candidate validation remain open; see the [checkpoint](checkpoint.md).

```python
import numpy as np
from openboost import NumericData, Problem
from openboost.binning import Binning
from openboost.stats import newton
from openboost.newton_order import rank, choose
from openboost.ops import partition

data = NumericData([[0], [0], [1], [1]], [10, 20, 30, 40], ("x",))
problem = Problem(data, [[0], [0], [0], [0]], data.row_ids)
binned = Binning(("x",), (np.array([.5]),)).transform(data)
fields = newton(problem, [-2, -1, 1, 2], [1, 1, 1, 1])
ranked = rank(binned, fields, reg_lambda=1, min_child_h=0)
selected = choose(ranked)
if selected is not None:
    left, right = partition(binned, None, selected.candidate)
    print(selected.candidate.key, selected.gain, left, right)
```

`rank` returns immutable `NewtonCandidate` records in descending exact gain order.
Each has an ordinary routing `candidate`, a `Fraction` gain, exact
`(left, right, parent)` sums in `candidate.names` order, and an evaluation identity
binding the fields, row selection, binning and configuration. Separate input
and configuration identities let growth validate actual node inputs and keep
score settings fixed across nodes. `choose` returns
the highest strictly positive record, or `None`. It accepts filtered/reordered
records from the same evaluation and rejects mixed bindings or duplicate keys.
Keep gains rational when comparing candidates, leaf budgets or layer totals.

Numeric thresholds, categorical equality conditions and both missing routes use
the supplied `BinnedData`. Unknown categories follow its missing-route encoding.
The operation never fits cuts or category dictionaries. Rows and field identity
must agree. G/H must be named once-weighted training fields with nonnegative
curvature on every row. Additional independent fields retain their own weights;
`min_information={"cohort": 2}` checks their exact sums without applying objective
weights. `min_child_h`, regularization and split penalty use their exact stored
binary64 values. Feasibility requires nonempty children and positive curvature.

Legal zero and negative gains remain in `rank` for callers composing a symmetric
layer. Candidate arrays contain correctly rounded, finite binary64 sums for
existing consumers, while `exact_sums` preserves every bit. An aggregate outside
that finite routing representation raises `ValueError`. Exact integer bin sums
can recover finite totals even when an ordinary intermediate sum would overflow.

The current implementation bins dyadic fields with integer arithmetic and uses
rational gain comparisons. All [three CPU growth policies](trees.md) accept
`ordering=rank` and preserve exact gains through node priorities and layer sums.
Leaf solving is a separate public operation. `leaf(fields, rows=None, reg_lambda=1)`
reduces named once-weighted G/H from original rows with exact dyadic arithmetic,
then rounds `-sum(G)/(sum(H)+lambda)` once to binary64. It does not multiply weights
again. The denominator must be positive and the returned value finite. Intermediate
sums may exceed binary64 when the final solution is finite; split routing records
still require finite aggregates. Pass `field_leaf=leaf` to any CPU grower, with
matching regularization configured alongside ordering.

This distinction matters: G=(1e16, 1, -1e16), H=(1,1,1), lambda=1 has an exact leaf
of -0.25, while a floating reduction can return zero. Exact split ordering alone
does not correct that leaf. The Normal CPU recipe now uses both operations by
default. Device execution uses the separate resident operations above. Measured
cost and remaining conformance/quality gates require their own evidence; arbitrary
scoring and leaf callbacks are unchanged.
