# Sprint 060: Current A12 structured Formula integration

Parent: 3ff97c3. Status: complete for structured integration.

## Plan and acceptance

1. Emit a separate frozen Formula packet with age/28 as structure and only
   composition features as tree inputs; preserve the ordinary GBDT packet.
2. Bind the public two-parameter saturation recipe with explicit structured
   inference, weighted direct parity and age/feature separation tests.
3. Run five frozen concrete folds under unchanged 90/30-second caps; retain
   exact replay, independent formula/score checks and raw artifacts.
4. Regression/lint/docs, reflection and local commit. A4 and CPU phase gates next.

No extrapolation/quality acceptance, global-formula parity or GPU claim.

## Results and reflection

The exporter now emits a separate formula-input packet without changing ordinary
GBDT inputs or the preprocessing freeze. Age/28 is structure, not a tree predictor.
Weighted final/best direct Formula parity and exact fresh replay pass. All five
real concrete folds pass, with independently recomputed saturation predictions
and scores, exact age separation and source IDs. See
[raw evidence](../benchmarks/v1/evidence/structured-060/README.md).

The first binding test incorrectly assumed raw feature width equaled encoded
width; train-fitted missing indicators make that false. Corrected the assertion
to verify removal of exactly the appended age column. The initial real export
lacked xlrd and failed before fitting; retry with cached ephemeral uv dependency
passed. Both failures are documented rather than changing data/recipe semantics.

Full CPU regression: 912 passed. No foundation production changes. A12 remains
open for interpolation/extrapolation quality, controls and complete searches.
A4 ranking is now the remaining current application adapter gap. Complete that,
then assess remaining A13/joint-search and D5/author requirements against B11;
do not infer CUDA readiness from recipe count. The Sprint 059 transition
checkpoint remains active. See
[learning](../learnings/2026-09-06-v1-structured-worker.md).

Closure: Ruff, strict MkDocs and whitespace pass. No push or publication.
