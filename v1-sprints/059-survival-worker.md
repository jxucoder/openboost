# Sprint 059: Current A10 fixed-scale survival worker

Parent: 20d9b7c. Status: complete for fixed-scale integration.

## Plan and acceptance

1. Verify weighted direct AFT parity with event/right-censored targets and reject
   malformed events/nonpositive times/unsupported scale or offset options.
2. Persist fixed sigma=1 and emit [log-time location, sigma] for evaluation;
   verify fresh replay and independently recompute censored likelihood.
3. Run all five frozen Veteran folds at unchanged 90/30-second caps, retain raw
   outcomes and source hashes. No licensing or quality acceptance inferred.
4. Regression/lint/docs, record remaining CPU-to-GPU prerequisites and commit.

No left/interval censoring, learned scale, test scoring, calibration or GPU claim.

## Results and reflection

Both direct A10 cases initially failed on missing support. The worker now binds
explicit event/right targets, fixed scale and weighted censored validation loss.
Final/best direct parity, fresh replay and invalid event/time/scale rejection pass.
All five frozen Veteran folds pass; independent erfc-based censored likelihood
agrees at rtol=1e-12/atol=1e-14. Source/output hashes match the
[raw evidence](../benchmarks/v1/evidence/survival-059/README.md).
Full CPU regression: 909 passed. No foundation production changes were needed.

## CPU-to-GPU checkpoint

The remaining current application adapter gaps are A12 structured Formula and A4
ranking; implement those next without adding new built-in objective families.
A13 real search integration, joint A9 selection and outstanding D5/independent
author checks remain unfinished CPU workflow/evaluation work. Apply the declared
B11/F1–F2 acceptance gates before reporting phase exit; test counts are not exits.
Then B12 implements device operations and recipe parity, and B13 verifies batched
train-many execution/cost. Full real quality and adoption remain required, but
must not become an invented requirement to polish every CPU performance path
before attempting CUDA. Any phase overlap requires an explicit plan amendment.

This slice validates fixed-scale survival integration only. Licensing closure,
IPCW/calibration and full A10 quality remain open. The next implementation is A12.
See [learning](../learnings/2026-09-06-v1-survival-worker.md).

Closure: Ruff, strict MkDocs and whitespace pass. No push or publication.
