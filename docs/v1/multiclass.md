# Multiclass and vector leaves

The CPU multiclass recipe grows one shared-topology vector tree per round.
All class directions use the same accepted raw snapshot and commit together.
ClassSchema preserves sorted class order through training and inference.

```python
import numpy as np
from openboost import ClassSchema, MixedData, Problem, RunContext
from openboost.recipes import multiclass

schema = ClassSchema.fit(["red", "green", "blue"])
x = MixedData([[0, "a"], [1, "b"], [2, None], [3, "a"], [4, "b"], [5, "c"]],
              range(6), ("x", "group"), ("numeric", "categorical"))
v = MixedData([[0.5, "a"], [4.5, "unknown"]], [10, 11],
              x.feature_names, x.feature_kinds)
train = Problem(x, schema.encode(["red", "green", "blue", "red", "green", "blue"]),
                x.row_ids, raw_width=3, classes=schema)
valid = Problem(v, schema.encode(["red", "blue"]), v.row_ids,
                raw_width=3, classes=schema)
fit = multiclass(train, valid, context=RunContext("multiclass-demo", 7), rounds=3)
model = fit.state.best_model
probabilities = model.predict_proba(v)
assert probabilities.shape == (2, 3)
np.testing.assert_allclose(probabilities.sum(axis=1), 1)
assert all(label in schema.values for label in model.predict_label(v))
assert len(fit.state.model.terms) == 3
```

Multiclass.geometry exposes weighted mean softmax loss and unweighted gradient
and diagonal bound arrays [N, K]. The bound is 2p(1-p), a diagonal upper bound
on the exact softmax Hessian, not the exact Hessian itself. The row-field adapter
applies original weights once. Initialization uses zero logits (uniform probabilities
before offsets), and every declared class must occur in training. Offsets enter
geometry once and remain outside accepted raw caches; supply inference offsets
explicitly. Fixed and backtracking steps use the common transaction machinery.

Vector components also work independently of classification:

- vector_newton binds gradient/diagonal-curvature arrays [N, L] to a problem.
- vector_score sums channel gains and charges one split penalty. vector_leaf
  solves each regularized diagonal direction. Configure both with the same lambda.
- vector_feasible requires positive child curvature in every channel and nonempty
  children, with an optional minimum child curvature.
- Each grower accepts separate leaf_fields from the same problem. A caller can
  supply projected split statistics while fitting leaves from full statistics.
  The caller owns the projection; this is not an automatic sketching policy.
- Tree.predict returns [N, L]. TreeTerm maps these outputs through an explicit
  [L, K] matrix and a scalar coefficient into model raw space.

The openboost-tree-v3 format stores vector payloads and rejects earlier formats.
Nested model artifacts preserve class labels and validate output-map dimensions;
loading requires no training objective. Numeric, categorical, missing and unseen
inputs use the same routing as scalar trees.

Independent tests cover all three growth policies, projected splits with full
leaves, three softmax rounds, offsets, rejected updates and fresh-process
probability/label persistence. These are correctness checks, not real classification
quality evidence. Full A6 multi-output workflows, specialized leaves, CUDA and
external-library quality/performance comparisons remain required work.
