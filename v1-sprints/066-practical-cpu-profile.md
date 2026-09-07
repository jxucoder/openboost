# Sprint 066: Practical CPU cost profile

Status: planned. Mapping: N2a / F1 / C4 / E1 and diagnostic cost evidence.
Depends on: [065](065-installed-run-isolation.md). Pin its actual revision at entry.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first check

Determine which repeated work and retained state limit realistic CPU execution
before changing the runtime. First verify that the harness actually terminates
an over-time and over-memory child and preserves its partial/error records.
A declared RAM field is not enforcement. Use a supported enforced environment
if the local host cannot provide the limit; do not label an advisory limit enforced.

## Work

- Freeze Housing fold zero without test labels: first 8,192 training and 1,024
  validation rows in frozen order, retaining and hashing their IDs. Reject absent
  subset sizes rather than silently shrinking them.
- Squared and Normal: 4/32/128 rounds, 32 bins, depth two, learning rate 0.1,
  regularization 1. Also run each at 32 rounds with 2,048 training rows: eight
  fixed-step cases. Add separate tiny deterministic rejection/backtracking fixtures.
- Enforce 120 seconds per case, two CPU threads and 8 GiB RAM. Stop further
  expensive cases after the first resource failure pending diagnosis; retain it.
- Separate uninstrumented end-to-end/peak-RSS measurements from instrumented
  stage and call-count profiles. Include preparation, prediction and diagnostics.
  Compare the small counting fixture from Sprint 063 without treating it as timing.

## Acceptance and reflection

Every attempted case has hashes, environment, cap enforcement, timing scope,
status and full-replay checks. Unattempted cases remain not_run with the stop
reason. A diagnosed resource failure is a valid diagnostic outcome, not a speed
pass. Record whether tree replay, encoding, hashing, histogram work or trace
retention dominates, and which measured cause [067](067-incremental-runtime.md)
will address. Do not change formal search budgets or implement an optimization
in this measurement sprint.

## Results

Not run. No practical timing or memory improvement has been established.
