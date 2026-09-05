# 2026-09-05: P7 matched-quality engineering value

## Context

P6 proves independent installed CPU/CUDA extensions, not useful speed, cost or
outside adoption. Measure the predeclared P7 tradeoff before broadening claims.

## Decision or Result

Freeze the P2 Housing resident configuration/seeds and compare four strategies
in fresh processes on one T4. Pair candidate timings with current legacy CUDA;
use frozen P2 for quality anchoring. Preserve negative value results. Independent
A+B+C changes the math, so it gets a separate quality/cost report.

## Changes

- Allowlisted value suite with both installed wheels, frozen data and P2 raw JSON.
- Four timed fits plus separate host profile and sampled device memory per cell.
- Evidence gate rejects incomplete/non-finite/fallback/missing-profile data;
  speed and quality failures remain booleans, never silently filtered rows.
- Document cache policy, unsupported eval, sampled-memory limitations and unknown
  billing. No core model or training changes.

## Verification

- CPU Housing seed 0: all four predictions repeat within tolerance; NLL
  1.094473772, CRPS 0.398590951, coverage90 0.967781008 agree with P2.
- Focused protocol + existing evidence runner tests: 25 passed before final
  path-evidence assertions; rerun below before commit.
- Real CUDA value measurements require a clean committed bundle next.

## Failed Attempts

- None on real hardware yet. Initial lint caught import ordering; corrected.

## Risks and Follow-ups

- cProfile captures host inclusive time, not a CUDA trace. Sampled device-wide
  memory is a lower bound, not an exact allocation peak.
- G4 not decided until raw matrix exists; G5 remains open without an outside
  author's actual attempt. Three Housing splits cannot establish broad adoption.

## Commits

- `15b3b4f` — preceding successful P6 GPU wheel evidence.

- Final harness verification: 25 focused tests passed; production/harness lint
  passed. Standalone CPU worker rerun passed after adding path profiling and
  partial-record checkpoints. A missing list bracket in the new profile code
  was caught by lint and fixed before any device execution.
