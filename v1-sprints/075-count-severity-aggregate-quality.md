# Sprint 075: A7 counts, A8 severity and A9 aggregate quality

Status: planned. Mapping: N4–N5 / B10–B14 / A7–A9 / R4 / E1, E3.
Depends on: the verified [071](071-real-multioutput-selection.md) selection pipeline.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first checks

Complete three separate selected-quality contracts, including actual joint
frequency-severity selection. First make the independent verifier detect double
exposure weighting, mismatched paid-event counts and incorrectly annualized totals.

## Work

- A7: count likelihood with exposure exactly once, distinct unit-exposure rate and
  period count predictions, Poisson deviance, aggregate totals and group diagnostics.
- A8: positive claim-level severity with documented eligibility and policy grouping,
  correct claim weights, Gamma deviance and appropriate Gamma/GLM baselines.
- A9: direct fixed-power Tweedie and linked frequency-times-severity predictions in
  matched aggregate/annualized units. Freeze the joint composition search on
  validation, not independently selected components relabeled as a joint optimum.
- Keep A7's event definition distinct from A9's matched positive-payment event
  count. Verify joins and zero/positive cases independently. Use the frozen power
  and full per-method trial budgets; do not expand the component Cartesian product
  without preregistering its total budget and method definition.
- Evaluate all five frozen folds, select before test release and replay both direct
  and composed inference in a fresh process. Include appropriate simple controls.

## Acceptance and reflection

A7, A8 and A9 each pass their own selected E3 deviance/units/weight contract.
Composition carries both component and selection hashes and correctly reconstructs
total predictions. Publish all failed trials, selected receipts, paired folds and
total composition/search cost. A7/A8 success cannot substitute for A9 acceptance.

Reflect on whether composition reuses generic artifacts/runtime and whether the
joint-selection protocol changes the conclusion from independently chosen parts.
Keep a newly discovered substantial composition fix separate from the evidence run.

## Results

Not run. Earlier component validation does not establish real joint selection.
