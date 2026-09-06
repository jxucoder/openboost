# Sprint 049: Full-fold Covertype validation integration

Parent: 0def101. Status: experiment complete; A3 validation failed its time budget. Mapping: Sprint 038 M3, A3.

## Plan and acceptance

1. Run the current multiclass worker on all five frozen Covertype folds with
   the existing four-round, depth-two, 32-bin, one-thread smoke configuration.
2. Retain failures and bounded process outcomes; verify seven-class probability
   shape/order/normalization, row identity and exact fresh-process inference.
3. Record source/input/model/output hashes and scope; update sprint/learning
   records and commit locally. Do not increase budgets silently to hide failures.

The existing 90-second fit and 30-second replay caps apply. This is full-dataset
validation integration, not a full quality grid, speed or GPU acceptance result.

## Results and reflection

All five full Covertype folds timed out at the unchanged 90-second fit cap.
No model/prediction artifacts were produced, so probability and fresh-inference
checks could not run. **0/5 passed; A3 full-dataset validation remains incomplete.**
Worker logs are empty; the process records show timeout/group termination, not
an algorithmic assertion or a measured bottleneck. Raw jobs, source/data/fold
identities and all process outcomes are retained in
[covertype-049](../benchmarks/v1/evidence/covertype-049/README.md).
Source and available artifact hashes match. No core/worker code changed.

This is a concrete cost counterexample to treating synthetic A3 tests as sufficient
consumer evidence. It is not a fair comparative speed result: the smoke budget
is fixed, no competitor ran, and memory was not capped. Source/preprocessing
verification precedes the timed worker; fit caps do not measure total export cost.
The full source has 581,012 rows and 54 features; all frozen folds were used.

Next prioritize a bounded profile of the same full-data input. Separate setup,
binning, initialization, per-round tree building, raw prediction and transactions.
Static inspection shows repeated prediction-time binning but does not establish
it as the cause. Preserve semantics and use measured evidence before optimizing
or increasing budgets. This redirects the next bounded slice, not required scope.
Other applications, D5, full searches, GPU and adoption remain required.

Latest CPU regression remains Sprint 048's 844 passes; this evidence-only slice
did not rerun it. Strict MkDocs and diff checks pass. See
[learning](../learnings/2026-09-06-v1-covertype-worker.md).
