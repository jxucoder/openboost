# Positive-payment frequency and severity

A two-model composition predicts positive-payment count rate times mean positive
payment. Counts must refer to the same eligible payment records as severity.
Raw claim counts that include zero payments are not interchangeable.

```python
import numpy as np
from openboost import NumericData, RunContext
from openboost.composition import paid_loss_problems, FrequencySeverity
from openboost.recipes import poisson, gamma

x = NumericData([[0], [1], [2], [3]], [0, 1, 2, 3], ("x",))
v = NumericData([[0.5], [2.5]], [10, 11], ("x",))
ftrain, strain = paid_loss_problems(x, [0, 2, 1, 3], [0, 6, 2, 15], [0.5, 1, 2, 1])
fvalid, svalid = paid_loss_problems(v, [1, 2], [3, 8], [1, 0.5])
frequency = poisson(ftrain, fvalid, context=RunContext("frequency", 7), rounds=3)
severity = gamma(strain, svalid, context=RunContext("severity", 7), rounds=3)
model = FrequencySeverity(frequency.state.best_model, severity.state.best_model)
output = model.predict(v, v, [1, 0.5])
np.testing.assert_allclose(output["period_mean"], output["annualized_mean"] * [1, 0.5])
```

paid_loss_problems binds declared policy aggregates to two problems:
Poisson paid counts with explicit exposure and business weights, and Gamma
positive policy-average payments with weights equal to business weight times
paid count. Zero-count policies remain in frequency but not severity. Counts
must be integers; positive counts require positive totals, and zero counts require
zero totals. At least one positive-weight paid policy is required for severity.
The helper uses the same predictors within each policy; it does not retain
claim-specific features or reconstruct raw joins.

The caller must filter eligible payments, aggregate by policy and handle
contradictory source records before constructing these inputs. The helper cannot
prove that supplied counts correspond to the supplied totals. Keep policy
entities in separate evaluation partitions.

FrequencySeverity accepts declared scalar regression models. Its inference
inputs may have different feature schemas but must carry identical policy row
IDs in the same order. Separate frequency_offset/severity_offset arguments are
applied to the respective raw models. Exposure scales paid count and period
loss once. Outputs name paid_count_rate, paid_count_mean, severity_mean,
annualized_mean and period_mean. Means are not calibrated aggregate distributions.

save/load use openboost-frequency-severity-v1, embedding both validated raw
models with fixed output roles. Corrupt/duplicate fields and invalid model
widths fail. The artifact is self-contained for inference; new exposure,
features and any offsets remain caller inputs. Model identities preserve which
dependency has each role, but cannot certify its training provenance.

The example selects each component by its own validation objective. Joint
selection by aggregate-loss quality requires an explicit workflow evaluation.
Tests verify three-round component training, products, units, row reordering,
aggregate rejection and fresh-process mixed-feature persistence. Real A9
quality/joins, joint selection, AFT and CUDA remain open.
