# Categorical features through shared operations

`MixedData` declares ordered feature kinds. Numeric columns accept finite values
or missing values; categorical columns accept homogeneous strings or integers.
Booleans, fractional category IDs and mixed token types within one column are
rejected. None/NaN represent missingness. NumericData remains the numeric-only
input record. MixedData owns immutable tuples; `.values` exports a detached object
array, so editing that export cannot alter the original data or its identity.

```python
import numpy as np
from openboost import MixedData, Problem, RunContext
from openboost.binning import Binning
from openboost.recipes import squared

x = MixedData([[0, "a"], [1, "b"], [2, "a"], [3, None]], [1, 2, 3, 4],
              ("value", "group"), ("numeric", "categorical"))
p = Problem(x, [[-2], [3], [-1], [2]], x.row_ids)
fit = squared(p, p, context=RunContext("categories", 7), rounds=2, bins=4)
unseen = MixedData([[4, "new"], [5, None]], [10, 11], x.feature_names, x.feature_kinds)
fitted = Binning.fit(x, bins=4)
assert fitted.categories[1] == ("a", "b")
assert fitted.transform(unseen).missing[1].all()
assert np.isfinite(fit.state.model.predict(unseen)).all()
```

The example reuses training/validation data for mechanics only. Real evaluation
requires the prescribed separate partitions.

Binning fits sorted unique dictionaries on training input only. Unseen tokens
follow missing routes; dictionaries do not expand during inference. An all-missing
training column has an empty dictionary and generates no split candidates. A
single observed category can split against missing rows. Binning identity includes
cuts, dictionary token types/order and feature kinds. Reusing codes from another
transformer fails identity checks.

Histograms retain physical counts and additive fields. Each categorical candidate
selects one dictionary value versus all other observed values, with both missing
directions considered. `Candidate.kind` exposes equality versus numeric threshold
semantics to callbacks. Depthwise, best-first and symmetric policies share those
candidates and original-row routing. This is not category subset search, ordered
target statistics or full CatBoost/LightGBM categorical parity.

The public names `Binning` and `Tree` replace NumericBinning and NumericTree;
there are no compatibility aliases. `openboost-tree-v3` persists typed dictionaries,
numeric cuts, explicit topology and missing routes. Old tree formats fail loading.
Inference validates feature kinds as well as names. Unknown tokens are allowed;
invalid dictionaries, out-of-range conditions and corrupt topology fail. Raw
ensemble artifacts embed this tree record, so no category fitting is needed on load.

CPU squared, Normal and Formula recipes accept mixed features through the same
preparation and learner path. This slice directly verifies complete squared
composition and all three mixed-feature growth policies; it does not establish
real classification or distributional quality parity. Specialized leaves,
CUDA and full evaluation remain required work.
[Multiclass and vector leaves](multiclass.md) now share this mixed-feature path.
[Binary classification](binary.md) now includes a verified mixed-feature recipe.
