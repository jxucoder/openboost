# Sprint 057: Bind matched paid events to frozen A9 rows

Parent: a041f1b. Status: complete for input binding; composition fit is next.

## Plan and acceptance

1. Independently reconstruct positive-payment counts/totals from retained claim
   rows and compare them with source aggregates, rejecting mismatches.
2. Map those aggregates to the exact frozen A9 training/validation policy IDs,
   verifying packet hashes, eligibility, target units and exposure weights.
3. Emit composition-only training/validation packets for all five folds; test
   mismatched counts, row identities, eligibility and weights before accepting.
4. Run regression/lint/docs, record raw manifests/reflection and commit locally.

This prerequisite slice proves input binding, not fitted composition quality.
The next slice trains and replays the public two-model composition on these
packets. The preparer reads the full source but opens no test-truth packet and
selects/scores no test partition. No frozen splits change.

## Results and reflection

All five frozen A9 training/validation bindings pass. Independent claim iteration
reproduces paid counts/totals exactly, while the fixture deliberately has different
raw claim counts. Policy order, eligibility, source exposure and annualized target
units agree with the frozen packets. Output zero-count/zero-total consistency,
positive exposure and at least one paid policy per partition were also verified.
The [manifest](../benchmarks/v1/evidence/paid-events-057/README.md) retains hashes
and source audit; large feature packets remain reproducibly generated artifacts.

Eleven focused tests pass, including altered aggregates, invalid claim joins,
eligibility, weights, targets, foreign/overlapping rows and test-label injection.
Full CPU regression: 892 passed. Ruff, strict MkDocs and whitespace checks pass.

The composition helper alone could not prove raw payment provenance. This binder
closes that input gap without changing the source freeze, population or foundation.
The summary and packet hashes establish reproducibility, not hostile-process
isolation. No model training occurred in this slice. Next train Poisson paid-event
frequency and count-weighted Gamma severity through paid_loss_problems, persist
FrequencySeverity, and replay annualized/period outputs on all five packets.
Component selection must be labeled separately from joint aggregate selection.
See [learning](../learnings/2026-09-06-v1-paid-event-binding.md).
