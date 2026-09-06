# Sprint 033: B10 frequency-severity composition

Parent: bf7ef2c. Status: complete for the bounded slice.

## Plan and acceptance

1. Build aligned Poisson/Gamma problems from declared positive-payment counts,
   totals and exposure; reject contradictory aggregates and preserve weight units.
2. Add a two-model inference artifact with fixed rate/severity/product semantics,
   aligned policy row IDs, explicit offsets and strict nested persistence.
3. Verify multi-round composition, policy reordering, exposure scaling, mismatched
   rows, aggregate errors and fresh-process mixed-feature round trips.
4. Run regression/lint/docs/build, reflect and commit locally.

The caller remains responsible for eligible-payment joins and dataset partitions.
This is A9 composition mechanics, not real quality or calibrated aggregate loss.
AFT and CUDA remain required. First failing check imports the absent composition.

## Results and reflection

Delivered paid_loss_problems for matched paid-count/positive-total policy
aggregates and FrequencySeverity for explicit two-model inference. Business
weights apply to frequency; severity averages receive business weight times
paid count. Policy IDs must align at inference. The versioned bundle embeds
both raw models and fixed output roles; Model.from_record reuses existing
strict validation without temporary files.

Nine focused tests pass: aggregate roles, three-round component training,
product identities, exposure scaling, policy-order invariance/misalignment,
contradictory aggregates, nested corruption/duplicates and fresh-process
mixed/missing/unseen inference. CPU regression: 693 passed. Ruff, strict MkDocs,
offline sdist/wheel and all sixteen installed-wheel examples pass on macOS/
Python 3.12.12/NumPy 2.3.5. Local wheel SHA256:
8b6610f3db3297e4e0c71744862cdf2cf1a989263c3233a650bfb99105e04615.

Observation: a raw scalar model alone does not carry the two-stage output roles.
Evidence: swapping dependencies changes bundle identity, and loading the bundle
restores every named prediction without training objectives. Decision: use a
small explicit composition artifact with embedded models and aligned row IDs.
The declaration does not prove model provenance or eligible-payment joins.

This completes the bounded A9 composition mechanics, not aggregate-quality
selection or real-data evidence. The helper uses policy-average severity with
shared predictors; claim-level covariates and external join auditing remain
workflow responsibilities. Component best models use their respective validation
objectives, not joint aggregate selection. Next is AFT/A10, including censored
target semantics and persisted scale/output transformations. Full A6 and
F0.3/F1–F5 remain open.

Status: complete for this bounded slice. See
[learning record](../learnings/2026-09-06-v1-b10-frequency-severity.md).
