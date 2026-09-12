# Exact resident scalar Newton ordering

This explicit operation orders scalar Newton candidates from their original
stored fields. Its historical exact-integer and fresh-inference checks are
described by the [checkpoint evidence boundary](checkpoint.md). The low-level
tree builder keeps its existing defaults; [Normal](normal-exact-default.md)
selects these exact policies by default.

```python
from openboost import device_newton_order

ordering = device_newton_order.rank(
    ops, candidates,
    reg_lambda=1.0,
    min_child_h=0.0,
    split_penalty=0.0,
    min_information={"cohort": 1.0},
)
split = device_newton_order.choose(ops, ordering)
if split is not None:
    left, right = ops.partition(candidates.histogram.rows, split)
```

The candidate batch binds prepared data, original routed rows and once-weighted
scalar gradient/curvature fields. Independent information columns retain their
own weight role. The operation decodes the original binary32 field bits and
accumulates signed integer statistics on the device. Candidate ordering and
strict gain positivity use exact rational cross products; curvature and named
information minima use exact sums. Physical row counts are separate. Exact ties
keep the first lexicographic condition.

`DeviceNewtonOrder` owns its integer statistics, child-score numerator and
denominator arrays and exact eligibility flags. Release it with `ops.release`.
It borrows its candidate/data/field/row binding, which must remain live for choice.
The chosen split borrows the candidate batch and composes with ordinary stable
partitioning. An optional candidate-bound `DeviceMask` passed to `choose` can
add caller constraints. It cannot weaken the exact Newton minima.

The context accounts for all retained and temporary device arrays. Construction
exports only status flags; choice exports one compact index. Explicit exports of
integer records are available for auditing. No host learner or numerical fallback
runs. Invalid domains, capacity failures and stale/foreign bindings fail explicitly.
The existing candidate constructor still imposes its documented finite aggregate
support. Exact leaf solving and changes to recipe defaults are separate work;
using exact order with an approximate leaf solver does not certify exact leaves.

The kernels scan original rows for each active candidate. Their full-workload
cost and candidate-specific conformance need separate validation. Historical
ordering failures remain retained in the original development history.
