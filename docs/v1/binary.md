# Binary classification with explicit class order

Fit ClassSchema on training labels only, then encode both training and validation
labels using it. Schemas sort homogeneous string/integer labels, reject missing
labels and require at least two classes. Validation may contain one known class;
unknown labels are rejected. Binary training requires both observed codes.

```python
from openboost import ClassSchema, MixedData, Problem, RunContext
from openboost.recipes import binary

schema = ClassSchema.fit(["no", "yes", "no", "yes"])
x = MixedData([[0, "a"], [1, "b"], [2, "a"], [3, None]], [1, 2, 3, 4],
              ("x", "group"), ("numeric", "categorical"))
v = MixedData([[0.5, "a"], [2.5, "unknown"]], [10, 11], x.feature_names, x.feature_kinds)
train = Problem(x, schema.encode(["no", "yes", "no", "yes"]), x.row_ids, classes=schema)
valid = Problem(v, schema.encode(["no", "yes"]), v.row_ids, classes=schema)
fit = binary(train, valid, context=RunContext("binary-demo", 7), rounds=3)
model = fit.state.best_model
probabilities = model.predict_proba(v)  # columns match model.classes.values
labels = model.predict_label(v)
assert probabilities.shape == (2, 2)
assert all(label in schema.values for label in labels)
```

Raw state remains `[N, 1]` logits. `Binary.geometry` exposes weighted mean logistic
loss and unweighted gradient/curvature. Signed-margin logaddexp loss and separate
sigmoid tails preserve small gradients for confident correct predictions. There
is no hidden curvature floor. The shared Newton adapter applies original weights
once; categorical/numeric trees and fixed/backtracking transactions are unchanged.

Initialization is the logit of clipped training weighted prevalence (clip=1e-6),
minus weighted mean offset. With heterogeneous offsets this is a deterministic
initializer, not an optimized intercept. Clip must be positive, less than 0.5,
and represent an upper probability below 1. Offsets affect objective geometry
once and remain outside accepted raw caches.

`model.predict` returns raw logits. `predict_proba` returns columns for the two
persisted class labels; `predict_label` decodes argmax, with exact ties selecting
the first sorted label. Supply inference offsets explicitly to these methods.
Probability inference uses an inference-only transform and needs no training
objective/data. Outputs are probabilities, not a calibration guarantee.

Class order is part of Problem/model identity and follows every accepted/best
snapshot. Mismatched train/validation schemas fail. The versioned ensemble format
now includes class labels; malformed/unsupported schemas and earlier formats fail
loading. Regression recipes reject classification-tagged problems rather than
silently treating class codes as numeric regression targets.

This slice verifies binary geometry and complete mixed-feature inference. Schemas
can describe multiple classes, but the current classifier model and recipe accept
exactly two; multiclass bounds, joint vector leaves and broader classification
quality evaluation remain next. No XGBoost/LightGBM/CatBoost parity or CUDA claim.
