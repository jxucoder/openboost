# Explicit resident tree numerical policies

`device_tree.depthwise` accepts two independent callback interfaces:

```python
from openboost import device_tree, device_newton_order
from openboost.device_newton_leaf import leaf

def ordering(ops, candidates):
    return device_newton_order.choose(
        ops, device_newton_order.rank(ops, candidates)
    )

tree = device_tree.depthwise(
    ops, data, split_fields, binning=binning,
    ordering=ordering, field_leaf=leaf, leaf_fields=leaf_fields,
)
```

`ordering(ops, candidates)` returns a live `DeviceSplit` from that exact batch,
or `None`. It owns scoring and feasibility, so it cannot be combined with
`scoring`, `legality` or nondefault numeric policy arguments. Empty children are
rejected before a tree is published. Exact ordering still uses the existing
candidate histogram construction and its supported finite-aggregate domain.

`field_leaf(ops, fields, rows)` receives the declared leaf fields and the actual
original-row view. It returns a live finite float32 buffer of `[output_width]`.
It cannot be combined with the histogram `leaf` callback or nondefault
`reg_lambda`. A callback can close over its own explicit parameters. This path
avoids leaf histogram construction, including for root-only trees where an
exact finite leaf can survive intermediate sum overflow.

Callbacks run in the node workspace. Returned values are copied into tree-owned
storage before scratch is released. Existing caller buffers are borrowed;
failed callbacks discard the partial tree and its new workspace. Scalar exact
Newton operations remain explicit; vector policies can use the same callback
interface with their declared output width. Defaults remain the existing
histogram-based score/mask/leaf behavior.

Historical checks exercise rational trees, failure ownership and fresh inference
for these callbacks. Their complete raw archives are outside this PR; see the
[checkpoint](checkpoint.md) for candidate validation and remaining acceptance.
The separate [Normal defaults](normal-exact-default.md) select exact callbacks.
