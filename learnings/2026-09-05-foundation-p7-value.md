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

## Developer materials

- Added a one-page objective/leaf/schedule cookbook and CPU/strict-CUDA capability
  matrix; corrected the older guide's CPU-only input/view description for CUDA.
- Added an unsolved independent-author task and observation form. It records
  actual time/help/private imports/core edits/GPU evidence and willingness to
  depend on OpenBoost. No author has attempted it and nobody was contacted.
- MkDocs build passed with existing griffe warnings. Core and plugin wheel
  hashes in the P7 bundle match P6; documentation changes do not alter that code.

## First T4 value result

- `eb61218`: [raw matrix](../benchmarks/results/foundation/20260905T183820Z-3c245f2d/README.md),
  3 passed / 0 skipped, 12 complete cells. Quality passes; default candidate
  warm median 2.078694 s versus legacy CUDA .149556 s: **13.899x**, budget fails.
- Independent A+B+C has worse NLL/CRPS despite coverage closer to nominal.
- Timings are uninstrumented. Separate profiles show sampling-thread calls and
  inconsistent inclusive parent/child times; do not use their percentages for
  causal attribution. Preserve them and collect isolated host profiles next.
- Sampled memory delta zero is an allocator/context observation, not zero peak
  memory. CUDA trace and exact peak remain gaps. No billing dollars inferred.

## Isolated profiling repair

- Split host profiling and memory sampling into separate fits. Add explicit
  synchronized inclusive objective/builder/session timers; their nested times
  overlap and synchronization overhead remains diagnostic-only.
- Add a 600-second, seed-0/four-strategy profile-only suite referencing the
  immutable original timing artifact/hash and checking unchanged quality.
- Focused tests: 27 passed. The new isolation test checks that the memory phase
  is absent from recorded profiler calls; Python 3.12 cProfile does not expose
  its monitoring state through sys.getprofile, so that initial test was replaced
  by direct recorded-call evidence. An incomplete-profile-matrix test first
  failed and now passes after enforcing exactly four seed-0 cells.
- Standalone CPU profile worker passed; production/harness lint and original
  matrix's offline validation passed. No core/plugin modifications.

## Isolated result and design review

- `9bb1ff3`: [isolated T4 artifact](../benchmarks/results/foundation/20260905T184856Z-dcd49569/README.md),
  3 passed / 0 skipped; four seed-0 profiles, quality matches parent. No original
  timing cells replaced. Core/plugin wheels still match P6 exactly.
- Default diagnostic fit 2.260 s: tree/session boundary 2.123 s, nested builder
  1.588 s, objective boundary .0885 s. Prioritize tree/validation/synchronization
  investigation; objective math is not the main measured cost. Do not remove
  contracts or infer promised savings from inclusive timers.
- Original performance verdict remains 13.899x, budget failed. No core speed fix
  or broader GPU capability expansion is included. Retain bounded research API.
- Developer guide and unsolved author task completed. G5 remains open; no outside
  author was contacted. Exact per-fit GPU memory peak and full CUDA trace remain
  unverified. Nominal coverage is overconservative; no calibration win claimed.
- P5 CPU regression (904 passed) and P6 CPU/GPU installation evidence apply to
  the identical core/plugin wheels. This slice changes only harness/docs; 27
  focused tests, production/changed-file lint and MkDocs passed. Both new GPU
  artifacts validate offline. Full hash/JUnit/privacy checks completed below.
- Final audit passed: every uploaded file and wheel hash, parent results hash,
  JUnit equality, unchanged executed implementation, private-path scan, and
  absence of memGetInfo from isolated host profiles. Final focused rerun:
  27 passed; production/harness lint and both offline evidence gates passed.
