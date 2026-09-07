# Sprint 076: A10 survival and A12 structured quality

Status: planned. Mapping: N4–N5 / B10–B14 / A10, A12 / R5, R7 / E1, E3.
Depends on: the verified [071](071-real-multioutput-selection.md) selection pipeline.
A10 additionally needs the provenance/source decision from 070; A12 is independent.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first checks

Evaluate selected models whose target and structural semantics differ from plain
regression. First independently detect treating a censoring time as an observed
event and leaking Formula's structural age input into its declared tree features.

## Work

- A10: resolve the Veteran original-source provenance or freeze a documented
  equivalent source before evaluation. Keep fixed-scale log-normal event/right
  censoring, censored NLL normalization and supported outputs explicit.
- Run all five frozen survival folds and full searches with appropriate AFT/simple
  controls. Report time/risk/survival outputs, applicable probability/ranking and
  calibration measures with their censoring assumptions; C-index alone cannot pass.
- A12: all five frozen Concrete folds, saturation Formula age/28 as structure,
  declared ordinary baseline features and appropriate structural/simple controls.
  Preserve separate real quality, interpolation/extrapolation and synthetic
  identification/misspecification checks, including independent Jacobian/directions.
- Use validation-only selection, fresh-process specialized inference and all failed
  trials. Do not generalize narrow AFT support to arbitrary interval censoring or
  a fitted structural hypothesis to causal identification.

## Acceptance and reflection

A10 satisfies its independent censored-NLL gate and applicable diagnostics;
A12 satisfies real selected quality plus required structural counterexamples.
Each has complete selected receipts, five-fold scores, prediction replay and cost.
An unresolved A10 source leaves only that application's source/quality cells open.

Reflect on whether typed targets and structural inputs remain natural public
components. Preserve negative identification and model-quality results rather
than simplifying the hypothesis after seeing test outcomes.

## Results

Not run. Existing bounded validation and synthetic mathematics are separate evidence.
