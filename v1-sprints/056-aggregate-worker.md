# Sprint 056: Current A9 annualized aggregate worker

Parent: d7622f4. Status: complete for direct aggregate integration.

## Plan and acceptance

1. Reproduce missing aggregate support with weighted public Tweedie parity.
2. Require explicit positive exposure weights for annualized targets, fixed
   evaluation power 1.5 and persisted annualized output metadata. Reject negative
   targets, missing weights, exposure inputs and second offsets.
3. Verify final/best selection, fresh replay and independent weighted objective;
   run five frozen aggregate folds at unchanged 90/30-second caps with raw evidence.
4. Regression/lint/docs, reflection and local commit.

This slice covers the direct aggregate path. Frequency-severity composition must
bind counts of eligible positive payments, not A7 raw claim counts, and remains
an explicit next slice. Neither path substitutes for full A9 quality/search.

## Results and reflection

Both direct A9 tests initially failed on missing support. The current worker now
uses public Tweedie at explicit power 1.5 with required positive exposure weights
and no second exposure offset. Final/best selection and fresh inference match
direct weighted recipes; malformed weights/targets and persisted power fail.
No foundation production code changed.

All five frozen direct aggregate folds pass with exact fresh replay. Independent
weighted objectives agree at rtol=1e-12/atol=1e-14. Validation weights exactly match
hashed period exposures; annualized targets times exposure reproduce paid totals.
Period predictions are retained alongside annualized outputs in the
[raw evidence](../benchmarks/v1/evidence/aggregate-056/README.md).

The direct aggregate consumer fits existing scalar geometry, but its units differ
from A7. Keeping that distinction explicit prevents silent double exposure.
This success does not validate a compound distribution, full A9 quality/search,
or the separate frequency-severity algorithm. Next bind positive-payment counts
to the same eligible totals and build the composition evaluation; then continue
A10/A12, unresolved A4, real searches and D5. All required scope remains open
until its own evidence passes. See
[learning](../learnings/2026-09-06-v1-aggregate-worker.md).

Closure: 881 CPU tests passed; Ruff, strict MkDocs and whitespace passed.
No foundation changes, push or publication.
