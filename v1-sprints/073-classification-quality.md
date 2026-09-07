# Sprint 073: A2 binary and A3 multiclass quality

Status: planned. Mapping: N4–N5 / B14 / A2–A3 / R1 / E3.
Depends on: the verified [071](071-real-multioutput-selection.md) selection pipeline.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Obtain selected real classification results with correct probability semantics
and honest full-input resource outcomes. First make the independent scorer reject
a permuted class mapping, non-normalized probabilities and a stale selected artifact.

## Work

- A2: all five frozen Adult runs with the official-test policy, training-only
  categories/missing handling, class metadata, log-loss and auxiliary AUC.
- A3: all five full Covertype folds, frozen class order, softmax normalization,
  multiclass log-loss and per-class quality. Use the declared full search spaces.
- Preflight practical resource caps using the same inputs. Earlier four-round
  completion and CPU histogram speedups do not establish 300/1000-round feasibility.
- Use supported XGBoost/LightGBM/CatBoost paths and per-method search spaces, fair
  size/quality comparisons, validation-only selection and fresh inference.

## Acceptance and reflection

Each application passes its own E3 median/worst-fold loss gate, class/weight/missing
semantics and replay. Publish all configurations, probability hashes, selected
receipts, per-class diagnostics, resource failures and total selection cost.

On a time/memory failure, retain the failed full workload and diagnose its actual
path before a new implementation sprint; do not reduce rows/rounds and reuse the
formal label. Reflect on whether practical multiclass execution is usable and
which remaining cost is intrinsic to the algorithm versus redundant foundation work.

## Results

Not run. Sprint 052's full-input short fits remain scoped CPU validation evidence.
