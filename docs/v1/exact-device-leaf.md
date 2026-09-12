# Exact original-row resident leaves

`openboost.device_newton_leaf.leaf` solves scalar Newton leaves from original
stored fields. The [checkpoint](checkpoint.md) distinguishes its historical
exact-bit/tree/inference checks from candidate validation. The low-level tree
builder keeps its defaults; [Normal](normal-exact-default.md) selects these exact
policies by default.

```python
from openboost.device_newton_leaf import leaf

value = leaf(ops, fields, rows, reg_lambda=1.0)
# value is an independently owned float32 [1] DeviceBuffer.
```

Fields and rows must be live records belonging to the same operations instance
and prepared data. Gradient and curvature must be named, once-weighted scalar
fields; curvature is nonnegative. The operation reduces the original stored
binary32 fields with signed integer arithmetic and forms the exact denominator.
It constructs no floating histogram. Intermediate sums may exceed float32 range
if the final rounded leaf remains finite.

The exact ratio `-sum(G)/(sum(H)+lambda)` is rounded once to binary32, with ties
to even. Exact zero is positive zero. A nonzero negative ratio rounded to zero
retains its sign. A ratio just above the largest finite value can still round
finitely; a ratio at or above the overflow midpoint fails explicitly. Zero
denominator, invalid domain or capacity loss also fails, without clipping or a
host numerical fallback.

The result owns its value independently of source lifetimes. Release it through
the execution context. All scratch is accounted; failed calls preserve caller
allocations. Only numerical status returns to the host during execution. This
operation can be used by an explicit tree leaf callback together with
[exact ordering](exact-device-newton.md). Normal integration, complete trajectory
conformance and performance each require their own validation scope.
