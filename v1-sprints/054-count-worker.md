# Sprint 054: Current A7 count/exposure worker

Parent: 003fdac. Status: complete; all five bounded real-data folds pass.

## Plan and acceptance

1. Reproduce missing A7 support with weighted direct public Poisson parity.
2. Bind strictly positive train/validation exposure as a structure role once,
   retain raw log-rate models, and require new exposure for period-count inference.
   Reject missing/invalid exposure, foreign-task exposure and extra offsets.
3. Verify weighted likelihood, final/best selection, fresh exposure-aware replay
   and count scaling. Run all five frozen real folds under unchanged 90/30-second
   caps and preserve failures without relaxing the workload.
4. Run regression/lint/docs, record evidence/reflection and commit locally.

No exposure-as-weight substitution, quality/search claim or CUDA claim. All
remaining application adapters and formal gates remain required.

## Results and reflection

The two new direct-parity tests initially failed on missing A7 support. The
adapter now binds exposure through public Poisson structure and retains explicit
period-count inference semantics. Weighted final/best recipe parity, fresh replay,
exposure scaling and invalid/missing/foreign inputs pass without core changes.

All five frozen frequency folds pass and replay exactly with positive means and
source IDs. Independent mean NLL checks differ by at most approximately 2.4e-14.
An initial relative tolerance of 1e-13 failed on fold two; the raw discrepancy and
math.fsum recomputation are retained. A 1e-12 relative/1e-14 absolute numerical
check passes; no prediction equality or quality threshold was relaxed.
See [raw evidence](../benchmarks/v1/evidence/count-054/README.md).

This supports the existing separation between raw state, exposure and original
sample weights. The frozen exporter calls exposure an offset; the public API
represents it separately and adds its logarithm once. No additional log offset
or exposure weighting is used. Five successful short runs do not establish A7
quality, calibration, full search acceptance or GPU readiness.

Next: A8 severity integration, followed by A9/A10/A12 and unresolved A4, real
search integration and D5 checks. No required application is dropped. See
[learning](../learnings/2026-09-06-v1-count-worker.md).

Closure: 866 CPU tests passed; Ruff, strict MkDocs and whitespace checks passed.
No foundation production files changed. No push or publication.
