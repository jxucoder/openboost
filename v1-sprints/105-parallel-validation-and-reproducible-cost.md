# Sprint 105 proposal: Parallel validation and reproducible cost evidence

Status: construction approved by the user's next "approve" after the Sprint 104
retrospective. Build the three slices below, committing each verified slice.
Hardware execution still requires its concrete source and budget freeze; no
new source upload, external baseline or author/model call has occurred.
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
