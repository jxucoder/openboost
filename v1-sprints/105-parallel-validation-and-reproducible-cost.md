# Sprint 105 proposal: Parallel validation and reproducible cost evidence

Status: proposed at the Sprint 104 retrospective boundary. No new hardware,
source upload, external baseline or author/model call is authorized by this card.
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
  with at most 10% regression on its 10,000-row counterpart. This is a proposed
  optimization gate, not a replacement for E4's external/real-data requirements.
- If deadlines or CPU reference cost prevent a required quality comparison,
  report the unresolved pair explicitly. Retain failed runs and profile overlap;
  never manufacture a speed ratio from the fastest partial repeats.

After this bounded cost response, return to required recipe construction in 080,
compatible train-many in 081 and real matched-quality evidence in 082. Author
friendliness remains deferred. All R1–R9/C1–C7/A1–A13 stay required.
