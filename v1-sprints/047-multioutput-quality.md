# Sprint 047: A6 standardized quality reporting

Parent: 8d8272e. Status: complete for A6 reporting semantics. Mapping: Sprint 038 M3, A6 reporting gap.

## Plan and acceptance

1. Reproduce missing standardized-average reporting for A6 quality artifacts.
2. Require hashed training rows/targets/scale for A6 comparisons; recompute the
   scale and reject row overlap, corruption and shape mismatches.
3. Report mean standardized RMSE alongside every original-unit target metric.
   Test scale effects, constant targets and an average hiding a target failure.
4. Run regression/lint/docs, record findings and commit locally. No new real
   quality claim or automatic E3 pass; model-selection provenance remains separate.

## Results and reflection

A6 reporting now requires a hashed train-row/target/scale bundle and recomputes
the frozen unweighted population scale. It rejects missing support, row identity
mismatches, training/evaluation overlap and scale changes. The standardized RMSE
is the mean of original-unit per-target RMSEs divided by training std; constant
target std is one. All per-target metrics remain mandatory and independently gated.

Five new tests exercise the formula, constant targets, normalization support and
a case where the standardized average improves but one target fails. The latter
still fails the A6 comparison. After fixing a fixture helper NameError, the initial
focused check reproduced the missing primary-metric support before implementation.
The full suite passes 832 CPU tests; focused artifact tests, Ruff and strict docs
pass. No new dataset runs or quality/performance claims were made in this slice.

Together with Sprint 046 this closes the scale-binding/reporting implementation
gap documented in Sprint 017. It does not close real A6 quality, selection
provenance, all recipes/devices, F0.3 or E3: E3_pass remains false by design.
Next remaining application adapters and real search integration, alongside D5.
See [learning](../learnings/2026-09-06-v1-multioutput-quality.md).
