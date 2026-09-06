# 2026-09-06: CPU recipe breadth is not complete foundation acceptance

## Context

After B10/AFT, inspect actual F1/B11 readiness before beginning GPU expansion.
Audited code revision b456bf3; this slice changes documentation only.

## Decision or Result

The public CPU subset passes 228 tests; total suite count 716 also includes
references/evaluation infrastructure. Complete A6 regression is absent, and
train-many does not yet share binning or implement validation-driven stopping.
Ordered parameter mutation, current external author wheels and all-application
integration remain open. F1/E0/E1/E2 cannot be declared complete.

## Changes

- Sprint 035 maps every A1–A13/R1–R9 and C1–C7 to code/tests and remaining work.
- Prioritized A6, prepared inputs/stop isolation/M32, ordered mutations/output
  dependencies, current installed author packages and real-workflow integration.
- Marked historical extension instructions at their local entry point.

## Verification

- Read public implementation/call paths, tests and planning contracts.
- Reproduced scalar squared rejection of two-output targets and absence of recipe
  early-stopping parameters.
- UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest
  tests/v1/test_public*.py -q --tb=short: 228 passed.
- Existing full suite evidence: 716 at the same production revision, Sprint 034.
- No runtime changes; no new benchmark, CUDA or real-quality evaluation.

## Failed Attempts

No attempted phase exit. Existing extension wheels were identified by retired
openboost.experimental imports and were not miscounted as v1 evidence.

## Risks and Follow-ups

The audit is an execution map, not a substitute for independent gate evaluation.
Keep all application scope and sealed held-out isolation. Next implement A6
multi-output squared workflows, then shared preparation and independent stopping.
See [Sprint 035](../v1-sprints/035-cpu-coverage-audit.md).

## Commits

Committed with the CPU coverage audit; parent b456bf3.
