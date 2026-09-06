# Sprint 046: Scale-bound current model selection

Parent: e83e7b5. Status: complete for scale-bound selection integration. Mapping: Sprint 038 M3, A6/A13.

## Plan and acceptance

1. Reproduce the judge accepting A6 without verified training normalization.
2. Require hashed row-aligned training targets and frozen scale, recompute the
   scale, bind inverse-standard-deviation metric weights, and report mean
   standardized RMSE rather than an arbitrarily normalized weighted score.
3. Execute a synthetic 16-configuration current A6 search, seal the independently
   recomputed winner and predict with that model in a fresh process after release.
4. Test missing/forged/misaligned scale inputs and winner selection, run regression
   and lint/docs, record evidence and commit locally.

All trial failures remain visible and prevent a search receipt. This is an
integration experiment, not frozen E3/E5/E7 evidence or OS-enforced isolation.
Real full-grid searches and remaining application adapters remain open.

## Results and reflection

The initial counterexample demonstrated that an A6 search lacking any training
scale binding could pass. The judge now requires hashed training targets aligned
to training IDs and a hashed scale artifact, recomputes the population scale,
and checks inverse-std coefficients exactly. Selection reports mean standardized
RMSE. Its old denominator preserved within-fold ordering under fixed coefficients
but gave a differently scaled numeric score; the definition is now explicit.
A hand-built test shows target normalization changing the correct winner.

Eight added tests cover missing scale binding, mean standardized scores, a
normalization-dependent winner and forged weights/scales/target rows/values/width.
The full CPU suite passes 827 tests. Ruff and strict docs pass. The new synthetic
current search runs all 16 distinct shared/independent configurations, records
all outcomes, independently selects openboost:15, seals/re-audits its receipt,
and predicts in a fresh process after feature release. Constant target output
remains exactly seven. Raw artifacts:
[current-selection-046](../benchmarks/v1/evidence/current-selection-046/README.md).

This closes the selection-coefficient binding portion of Sprint 017's A6 gap;
final comparative quality reporting is still open. Hashes and row alignment
establish consistency with the trusted supplied training data, not its external
provenance. The original objective remains open: real complete searches across
all applications, test isolation, D5, GPU and author/adoption gates. This synthetic
run is neither formal A13 cost evidence nor 16 configurations on real data.
See [learning](../learnings/2026-09-06-v1-scale-bound-selection.md).
