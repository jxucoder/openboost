# 2026-09-07: Measure practical execution before extending CUDA coverage

## Context

The user asks whether OpenBoost is fast after the successful correctness rerun.
Existing tiny GPU fits use hundreds of dispatches/synchronizations, and the prior
real CPU A6 probe is expensive. Neither establishes competitive GPU speed. The
user explicitly approves an early performance checkpoint on squared and Normal.

## Decision or Result

[Sprint 104](../v1-sprints/104-early-performance-checkpoint.md) moves a bounded
same-host CPU/CUDA measurement ahead of new recipe ports. Use synthetic workloads
at 10,000/100,000 training rows, plus a 1,000-row Normal control. Require complete
measurements and comparable independent held-out metrics before reporting a ratio.
Retain timeouts instead of replacing the CPU reference or reducing its work after
execution. Formal E4, real application value and external baselines remain open.

## Changes

- The measurement child records dataset creation, context/preparation, training,
  export/cleanup, model loading and supported CPU inference. It writes progress
  after each fit so a later timeout preserves completed repetitions.
- CPU uses two fits, CUDA four, with fixed inputs/settings and fresh per-case
  processes. A separate first/warm profile records inclusive compiler, dispatch,
  synchronization and blocking-transfer costs; these overlap and are not official
  timing ratios. GPU children receive separate disk-cache paths.
- A failure-closed quality judge independently recomputes scores from saved raw
  predictions, verifies state/repetition/source identity and withholds ratios
  on incomplete work, profiling, leaked buffers or quality changes beyond the
  declared thresholds. Constructed judge controls are not GPU evidence.
- The existing bounded Modal runner and isolated CPU builder are reused without
  modifying previous frozen sources. Run 10 uploads 46 exact files, requests one
  T4/two CPUs/8192 MiB with 900/600-second function/test limits and zero retries.

## Verification

- Sixteen focused local tests cover actual CPU model replay, independent Normal
  scores, workload identity, malformed/quality-failing judge inputs and a real
  child timeout with retained partial work.
- The exact source closure builds as an offline wheel and collects six GPU test
  IDs in isolation. Local collection executes no CUDA case.
- Full CPU regression passes 1965 tests with one Linux-only skip in 13.96 seconds.
  The sixteen focused checks pass in 0.80 seconds. Production/support Ruff and
  MkDocs pass, with the existing 090 evidence-link warning. Source-freeze and
  approved dispatch guards pass with the fixed output absent.

## Failed Attempts

The [unqualified development calibration](../v1-sprints/104-local-calibration.json)
needs 25.59–26.14 seconds
per twenty-round Normal fit on 1,000 rows (four repetitions). Source inspection
shows per-row Python interval comparisons; this is a hypothesis for cost, not a
profiled attribution. It motivates explicit deadlines and a small Normal control.
No performance ratio or formal benchmark claim uses this development observation.
Its 31 production-source hashes match `c84d566`; the evolving benchmark script
was not frozen, and that provenance limitation is explicit in the retained record.

Call-path inspection catches explicit GPU binning being combined with a nondefault
bin-count argument, which DeviceRun rejects. The harness now lets the supplied
binning own its configuration. This is corrected before source freeze or upload.

## Risks and Follow-ups

Larger public CPU Normal fits may time out, leaving their speed ratios unavailable.
The profile may also exceed its separate cap; preserve that failure. First/warm
differences include more than compilation. CPU prediction of a CUDA-trained model
does not establish general GPU inference throughput. Tiny/missing/weighted fixtures
and internal CPU/CUDA quality are not real-task or external-library evidence.

## Commits

- `c84d566` closes the preceding bounded CUDA comparison coverage.
- This slice freezes the user-approved early performance checkpoint for one run.

## Automatic approval review after the freeze

`8677bba` commits the locally verified packet. The attempted dispatch is rejected
before process creation by automatic approval review. Its reason distinguishes
approval of the performance study from authorization to upload the exact private
repository payload to Modal. No output directory or remote execution exists and
the tenth allowance is not consumed. No alternate upload or indirect retry occurs.

Restore only upload authorization to pending; the study/compute approval remains.
Preserve all 45 frozen source hashes, cases, budgets and the original local
collection report. The next user approval must explicitly cover the 46-file,
approximately 358 kB Modal upload for the bounded single T4 invocation. This is an
automatic-review requirement, not a new mathematical or benchmark gate.
The local recheck confirms all 45 prefrozen hashes remain unchanged, the output
is absent and the dispatch guard rejects pending upload authorization.

## Explicit authorization on September 8

The user replies "approve" to the exact 46-file, approximately 358 kB Modal
upload and one T4 invocation with two CPUs, 8192 MiB, 900 function seconds,
600 test seconds and zero retries. Restore only upload authorization to approved;
retain the unchanged source freeze and historical collection report. This resolves
the automatic review's stated missing authorization without an alternate transfer
mechanism. Record a clean execution revision before the single invocation.

Verification: 73 focused performance/aggregation/Normal/comparison manifest tests
pass in 2.12 seconds. All 45 prefrozen hashes match; the exact upload contains
46 files and 357,936 bytes. The fixed output is absent and the approved dispatch
guard passes. No production or benchmark source changes accompany authorization.

## Run-10 result and retrospective

The exact authorization is committed at `c8f7ebc`; the one Modal invocation then
completes with one pass and five failures. All 46 uploaded hashes, 31 installed
core sources, eighteen pinned packages and sixteen case artifacts match. The
[archive](../benchmarks/v1/evidence/early-performance-104/README.md) preserves the
false verdict, all child timeouts and partial completed repetitions. The tenth
allowance is consumed without retry. Production and frozen benchmark code do not
change; public documentation is updated after execution to correct stale Normal
failure claims and describe the measured performance boundary.

Squared 10,000 rows is the only qualified pair: 7.044 s CPU versus 2.941 s warm
GPU, ratio 2.395 with relative half-MSE difference 1.09e-7. At 100,000 rows GPU
retains a 20.486 s first fit and two roughly 13.75 s warm fits before timeout;
the third warm fit and final quality artifacts are absent, so no ratio qualifies.
Normal GPU warm fits complete at 5.111/6.858 s for 1,000/10,000 rows; CPU pairs
time out. The 1,000-row CPU first fit takes 48.009 s, substantially above the
unqualified local calibration. This exposes insufficient remote deadline headroom.
Normal 100,000 rows retains only a 49.503 s GPU first fit. Its separate profiled
first fit takes 57.953 s; the required warm profile is missing after the 60 s cap.

Normal 10,000/100,000-row fits make 9,417 launches and 12,982 synchronizations.
The partial profile attributes 28.964 inclusive seconds to blocking exports and
22.766 seconds to launch, including 15.694 seconds in compiler calls. These
overlap; blocking exports include preceding kernel wait. Source inspection finds
serial validation/reductions and full-row scans per histogram cell. A specific
dominant kernel is not measured. The
[105 proposal](../v1-sprints/105-parallel-validation-and-reproducible-cost.md)
starts with parallel boolean/domain validation, preserving numerical reduction
order, public checks and accepted/best-state semantics. Further hardware requires
a new frozen allowance. Stop here for retrospective; all formal scope remains.

## Offline regeneration counterexample

The first offline auditor correctly rejects its assumption that the same NumPy
version regenerates exact remote Problem hashes. On macOS x86_64/NumPy 2.3.5,
training and validation identities differ from Linux for all five fully saved
children. All five saved models replay bit-exactly on local validation features;
local metric differences are at most 4.74e-9. The exact input discrepancy remains
unisolated, and replacing float32 sine with double sine does not recover it.
Do not silently relax the frozen judge or replace remote targets. Its qualifying
pair has matching same-host CPU/GPU inputs and independent score recomputation.

The offline audit now distinguishes source/hash/state/prediction verification
and qualification from retained same-host metrics from host-dependent input
regeneration. It records the Mac observation explicitly. A check on another host
still runs prediction replay but does not require its regeneration observation to
equal the recorded Mac observation. Future evidence must preserve exact input
bytes, and completed models/quality after each fit, to survive later timeouts.

## Closure verification

- The offline run-10 audit verifies 46 dispatch source hashes, nineteen raw artifact
  hashes, all six case identities, all 22 retained finished fits and five exact
  saved-model prediction replays. It reproduces eligibility from the retained
  same-host scores while reporting the local input-regeneration gap separately.
- The prior run-9 combined audit still passes and verifies all 420 indexed run-8
  files unchanged. The run-10 active dispatch guard rejects reuse; all 45 prefrozen
  payload hashes match the historical dispatch manifest. Only current public
  documentation changes after the run, not production/benchmark implementation.
- The 73 focused performance/aggregation/Normal/comparison manifest tests pass in
  2.62 seconds. Production/analyzer Ruff and MkDocs pass, with the existing 090
  evidence-link warning. The earlier 1965-test CPU regression remains the last
  full suite; no production code changes in this closure.
- The archive index hashes all 23 other archive files. Sixteen retained JSON
  artifacts total 4,110,926 bytes, below the frozen 32 MiB limit. Raw timeout/log/
  verdict bytes are preserved, including literal whitespace.

`c8f7ebc` is the clean execution revision; this closure commits evidence and the
retrospective proposal without pushing or starting another hardware run.
