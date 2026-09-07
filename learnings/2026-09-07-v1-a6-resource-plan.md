# 2026-09-07: Make the full A6 OpenBoost resource matrix explicit

## Context

Protected selection has only exercised four-round synthetic configurations. The
next real preflight must retain the frozen 300/1000-round grid and account for both
shared and independent topology without hiding extra trials.

## Decision or Result

Compile each topology as its own 16-configuration method from the frozen numeric
XGBoost-shaped tree grid, retaining all values. Apply frozen binning/patience and
five fold seeds. This prepares the OpenBoost portion only; comparators remain open.

## Changes

- A non-executing plan compiler emits 160 jobs, resource ceilings and first probes.
- Input file hashes bind search, Parkinsons source and preprocessing freezes.
- The generated plan is recorded in v1-sprints/070-a6-resource-plan.json.

## Verification

Seven focused tests pass: complete config preservation, unique jobs, full budget
accounting and rejection of omitted/duplicate/shortened/unknown configurations or
changed caps/folds. Lint passes. No model cost or real quality is measured here.

## Failed Attempts

No full search was attempted. The 80-hour worker ceiling and 160 reserved CPU-hour
ceiling exclude setup, inference and comparators; they are upper bounds, not forecasts.

## Risks and Follow-ups

Bind verified real packets, then run the two named first probes sequentially with
stop-on-failure. Preserve test material outside candidate containers. Comparator
coverage and protocol-derived complete scope cannot be inferred from this submatrix.

## Commits

- OpenBoost A6 resource preparation; parent `68d7b0d`.

Final validation: **1038 CPU tests passed, 1 Linux-only test skipped** with two
test workers. Production/support lint and documentation build pass.
