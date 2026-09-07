# 2026-09-07: Retain A6 comparator coverage in the CPU search plan

## Context

The completed paired fit closes the measured scoring detour, but the existing
resource plan covers only 160 OpenBoost jobs. It cannot establish complete A6
candidate/comparator scope or support a complete search cost ceiling.

## Decision or Result

Keep the original resource plan unchanged. Add a separate evaluator-derived plan
for five methods, five folds and sixteen configurations: 400 fits. The existing
baseline paths are XGBoost shared multi_output_tree, LightGBM independent target
models, and CatBoost shared MultiRMSE; OpenBoost has shared and independent modes.
These choices describe existing adapter paths, not exhaustive incumbent capability.

The maximum sequential worker allowance is 720000 seconds (200 hours), with
1440000 reserved CPU-seconds (400 CPU-hours). This is a timeout ceiling, not an
expected runtime or total cloud cost. Startup, replay, selection, train-many
schedules and GPU work are excluded.

## Changes

- `a6_search_plan.py` compiles all methods and validates producer plans against a
  separately trusted design. Recomputed counts do not permit omitted methods,
  folds, trials, changed parameters, budgets or a fabricated dispatch-ready state.
- `v1-sprints/070-a6-cpu-search-plan.json` retains every trial and pins the design,
  source/preprocessing freezes, baseline worker, dependency lock and compilers.
- The audit found that baseline_worker does not accept explicit bins=255. Planned
  comparator jobs include this requirement and remain undispatchable until its
  translation is implemented and verified. Existing native defaults are not
  asserted to satisfy the shared bin budget.

## Verification

The initial full-method assertion failed against the original 160-job planner.
Eighteen focused tests pass across the original and new planners. Regression
suite: 1106 passed, one Linux-only skip. Lint and documentation build pass. No dataset export, upload or remote execution occurred.

## Failed Attempts

No execution attempted. The missing explicit bin translation is a discovered
adapter gap; it is retained as a blocker rather than silently omitting comparators.

## Risks and Follow-ups

The design and plan pins must remain evaluator-owned. This is not an integration
with the full R/C/A/E ledger or the selected-test release gate. Next implement and
verify explicit comparator bin translation, including installed fit/reload and
stopping checks, before further full-budget resource preflights. Independent
accounting/isolation remains Sprint 069 work. No full matrix is launched.

## Commits

This verified slice follows `2ccc8ec`; no push.
