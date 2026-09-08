# Sprint 105: Parallel validation and reproducible cost evidence

Status: construction and the separately approved run 11 complete. All 474 T4
checks and all three frozen cost gates pass at clean `dd84247`; all 28 retained
fits replay from exact input bytes. Keep the optimization and stop at the planned
retrospective below. All eleven GPU allowances are consumed. No external-library
baseline or author/model call is part of this sprint.
Mapping: B12 / F3 / shared CUDA foundation cost; formal E4 remains in Sprint 082.

## Why this slice

[Run 10](../benchmarks/v1/evidence/early-performance-104/README.md) establishes one
qualified squared 10,000-row internal warm speedup, but four incomplete timing
pairs and an incomplete profile. It shows thousands of launches/synchronizations
and substantial blocking export wait at 100,000 rows. Source inspection finds
serial all-row validation; removing its serial scan is a concrete candidate that
does not require changing floating-point reduction order or algorithm decisions.
It is not yet a measured dominant-kernel diagnosis or a promised speedup.

## Construction order

1. **Exact evidence inputs and partial results.** Retain lossless generated input
   bytes plus identities, and persist the finished model, predictions and task
   scores after every completed fit. Verify exact offline replay from those bytes
   in an isolated CPU installation. A later timeout must preserve earlier complete
   fit artifacts while the declared multi-repeat case still fails. First test a
   generator/target mismatch and a real interrupted later repetition.
2. **Parallel validation with the same public behavior.** Start with finite/domain
   and nonnegative field checks, then routed-row bounds/duplicate checks only if
   the first change helps. Use deterministic boolean/integer flag reduction;
   check every row, including zero-weight rows. Keep public input errors,
   ownership, rollback, same-stream ordering and synchronization at observable
   error boundaries. First test invalid values in the final chunk, empty/tail
   chunks and duplicates across blocks. Retain a small-input path if evidence
   justifies it. Do not remove validation or change numerical histogram/loss sums.
3. **One feasible, frozen comparison.** Select a narrow correctness and cost matrix
   with unchanged recipe settings and exact stored inputs. Capture per-kernel-name
   counts and separate CUDA-event/profile evidence for validation; official timing
   stays unprofiled. Compare the frozen original and candidate on the same device
   and input bytes, with complete repetitions and quality checks. Freeze sources,
   resources and deadlines before requesting a new invocation. The 50 s/60 s
   deadlines from run 10 are consumed, not retroactively extended.

Commit and reflect after each verified slice. Stop if the measured bottleneck is
elsewhere or the optimization complicates the public boundary without meaningful
benefit. Then choose a specific histogram/reduction or CPU comparison response;
do not expand into an unbounded CPU speed contest.

## Acceptance and eval

- Exact input/model/prediction/score replay from retained bytes, with a test that
  detects a changed target even when predictions are unchanged. A partial run
  cannot receive a complete-case ratio.
- Real-device finite/nonnegative and row-validation equivalence, including
  invalid zero-weight rows, block boundaries, duplicate indices, bounds and
  injected allocation/dispatch failures. No skipped GPU case counts as passing.
- Preserve relevant scalar/Normal/D2 public behavior, split/leaf/prediction/task
  metric parity and buffer cleanup. Boolean scheduling changes must not alter
  accepted or best-state decisions; no tolerance relaxation.
- Same-host first and three-warm measurements. A provisional engineering target
  is at least 20% lower complete warm fit cost on the selected 100,000-row case,
  with at most 10% regression on its 10,000-row counterpart. This is the approved
  optimization gate, not a replacement for E4's external/real-data requirements.
- If deadlines or CPU reference cost prevent a required quality comparison,
  report the unresolved pair explicitly. Retain failed runs and profile overlap;
  never manufacture a speed ratio from the fastest partial repeats.

After this bounded cost response, return to required recipe construction in 080,
compatible train-many in 081 and real matched-quality evidence in 082. Author
friendliness remains deferred. All R1–R9/C1–C7/A1–A13 stay required.

## Slice A: exact inputs and recoverable fit evidence

`benchmarks/v1/performance_evidence.py` stores owned input arrays as lossless
compressed bytes, with array checksums, complete Problem identities and one
snapshot binding. Loading never calls the generator. Unsupported structured or
categorical problems are rejected explicitly; this format serves the frozen
squared/Normal cost probes, not general model persistence.

Each finished fit records its model, exact predictions, independent scores,
state/ownership counters and CPU inference before the next fit starts. Fit timers
exclude this post-fit audit/serialization work; child wall time includes it.
All saved fits are verifiable after timeout, while an unfinished required repeat
set stays incomplete. Neither this new format nor these CPU tests rewrite run 10.

Ten new tests and sixteen existing checkpoint tests pass (26 total, 1.24 s).
The new tests include actual child termination after one completed fit, changed
targets despite identical predictions, forged array/problem hashes, and invalid
complete-report counts. The
[isolated CPU check](105-input-replay-local/verification.json) replays two Normal
fits with the generator disabled, installed core/NumPy only and no CuPy/Numba.
Its commands and exact source hashes are retained in the neighboring installation
record. No GPU execution or performance claim is made. Next: parallel field
validation, keeping public errors and existing allocation/launch contracts.

## Slice B: cooperating field-validation blocks

The candidate assigns one 128-thread block to each field. Lanes scan disjoint
strided rows and all participate in `cuda.syncthreads_or`, including lanes with
no rows. One lane writes the same int32 flag as before. No scratch allocation,
extra launch, floating-point reduction, error suppression or ownership change is
introduced. Resident row-index validation is left unchanged until field-validation
benefit is actually measured, as required by the construction order.

Thirty-nine new real-CUDA cases collect locally: float32/float64, nonnegative on/
off, empty/short/exact/tail shapes up to 100,003 rows and seventeen columns,
per-column flags against an independent NumPy predicate, invalid zero-weight
tails, negative zero, and allocation/before-launch/after-launch recovery. Existing
aggregation/Normal/D2 consumer tests remain unchanged. No GPU case has run here.
The next slice freezes the original/candidate installations, exact inputs,
consumer checks, per-kernel instrumentation and feasible cost deadlines.

Full local CPU regression: 1975 passed, one Linux-only skip, 14.98 seconds.
Production and new test lint pass. This remains an unvalidated CUDA candidate
until the next real-device result; no performance gate has passed.

## Slice C: bounded original/candidate comparison

The [concrete request](105-validation-run11-request.md) freezes 88 source files,
474 collected cases, seventeen retained artifacts and one T4 invocation with
900-second function / 600-second pytest caps and no retry. Upload and invocation
are pending. Three cost cases use exact saved inputs, complete per-fit evidence
and verified original/candidate installations. Squared CPU controls remain
mandatory; Normal is scoped to same-algorithm GPU equivalence and cost.

The test selection uses prior observed durations: the omitted 96-case Normal
runtime matrix consumed 51.9 seconds by itself, while selected existing cases
used about 69 seconds. The 485-second child budget and thirty-second installation
cap leave 85 seconds for correctness, setup and audit. This corrects run 10's
inadequate per-case deadlines without altering its consumed freeze or verdict.

The actual local baseline builder initially failed because a nested environment's
`--system-site-packages` does not inherit its parent's virtual dependencies.
Appending the parent's resolved installed-package paths after the baseline core
fixes the issue; exact source and version checks still apply. The
[local result](105-baseline-install-local.json) proves the 31 original source
hashes and two saved CPU replays, with no CUDA/package-18 claim. The isolated
candidate snapshot collects all 474 frozen cases. Full CPU regression passes
1994 tests with one Linux-only skip in 12.95 seconds.

## Reflection after three construction slices

Slices A (`680bf84`) and B (`8db2aef`) are committed. This third slice completes
the measurement harness and concrete request. The only production change remains
the field-validation scheduling; no numeric algorithm or tolerance changed.
The hypothesis is now falsifiable by exact models, complete repetitions and
separate validation-operation measurements. Local correctness of the evidence
format is established; CUDA correctness, bottleneck significance and speed remain
unknown. Keep row-index validation unchanged until measured benefit justifies it.
After the bounded real-device result, reflect and choose whether to keep this
optimization or return to required CUDA recipes/train-many and formal cost work.
Author friendliness remains deferred and every R/C/A family remains required.

## Run 11 result and sprint closure

The user approves the exact packet after `2101fd7`; approval is committed at
`dd84247`, which executes once. All 474 cases pass, with all 88 source hashes,
31 installed candidate and 31 original core hashes, eighteen package versions,
installed D2 and the separate CPU environment verified. All seventeen declared
JSON artifacts are retained within 64 MiB. The
[complete evidence](../benchmarks/v1/evidence/parallel-validation-105/README.md)
preserves raw verdict, models, exact inputs, timings and the offline audit.

Warm original/candidate fit medians are 2.838/2.749 seconds for squared 10,000 rows,
13.513/8.947 for squared 100,000 rows and 6.538/5.665 for Normal 10,000 rows.
All GPU models/predictions match exactly across both arms and repetitions. The
large squared fit takes 33.79% less time, exceeding the frozen 20% target; both
smaller cases improve and pass their regression limits. Squared CPU controls
also complete with relative half-MSE differences below 1.1e-7. All 28 completed
fits replay exactly from retained bytes on macOS; no generator reconstruction is
used. The standalone validation-operation interval improves about 30x, but its
host enqueue/wait gaps and separate instrumentation forbid a fit-speed claim.

Worker time is 380.143 seconds, pytest time 377.754 seconds and total dispatch
765.509 seconds including image construction/setup. All child deadlines pass;
there is no retry. The raw run-10 failure and all 450 indexed files from the
previous three archives remain unchanged.

### Retrospective and next construction proposal

The hypothesis is supported within this scope: cooperative boolean validation
reduces complete large-fit cost without changing mathematical decisions or adding
launches. Keep it. Exact inputs and per-fit evidence also solve the previous
reconstruction/partial-record weaknesses for new runs. The first offline analysis
attempt incorrectly normalized slashes inside parameter IDs; it now normalizes
only the test module path, matching JUnit. No device result or test was changed.

The remaining large squared cost is 8.947 seconds, with unchanged 4,531 launches
and 6,272 synchronizations. This is not proof of the next bottleneck. Do not turn
this successful optimization into an open-ended kernel campaign. The next useful
foundation test is more required applications through the same components:

1. Resume 080 with binary and exposure-aware Poisson objective operations. Freeze
   independent base/loss/gradient/curvature expectations first; then compose the
   resident fields/tree/runtime operations. Check extreme margins, offsets,
   zero/nonuniform weights, failure ownership and explicit unsupported inputs.
2. Carry classification/survival inference metadata through the device export
   boundary, then complete multiclass, fixed-scale event/right-censored AFT and
   independent/shared multi-output squared cells. Current `DeviceRun.export`
   exports raw base/terms only; correct output interpretation and fresh CPU
   persistence are required use-case behavior, not an optional wrapper.
3. Freeze and execute affected device parity only after each concrete construction
   packet is reviewable. Then enter 081's actual compatible M=1/8/32 batching,
   followed by 082's real-data, multi-seed, matched-quality external cost gates.

Acceptance stays intermediate and final E1 parity, same-ID state/ownership,
specialized inference round trips and explicit fallback rejection. Do not count
Python loops over GPU fits as batching, these synthetic ratios as E4, or this
bounded Normal subset as full conformance. All R1–R9/C1–C7/A1–A13 remain required;
author friendliness stays deferred. Stop here for the planned retrospective;
no new implementation or hardware allowance follows automatically from this run.
