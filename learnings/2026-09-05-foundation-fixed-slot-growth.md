# 2026-09-05: Remove split compaction from fixed-slot growth

## Context

P7 found a 13.899x default GPU fit regression. The isolated host profile places
most time in LevelWiseBuilder and its session boundary. Before relaxing any
correctness checks, eliminate variable-length indexing inside fixed-slot growth.

## Decision or Result

Use fixed-shape selections for tree arrays. A nonroot slot enters the next
frontier iff its fixed parent split in this round. Preserve terminal leaves;
root must never reenter through the clamped parent index. No numerical split,
histogram, leaf, validation or cached-prediction contract changes.

## Changes

- LevelWiseBuilder uses where and parent-index gathering instead of boolean
  extraction/scatter of split arrays; parent mapping is created once per tree.
- Added branching/early-leaf tests that reject boolean compaction and compare
  the entire tree/predictions against the existing original-row oracle.

## Verification

- Both new tests failed on the original boolean extraction before the edit.
- Focused CPU builder/split/leaf/objective/dispatch regression is recorded below.
- Real CUDA trainer conformance and matched-quality P7 measurements follow a
  clean implementation commit; no speedup inferred from fewer operations.

## Failed Attempts

- None on hardware yet. This is a bounded candidate, not a proven performance fix.

## Risks and Follow-ups

- All scalar checks and independent prediction traversal remain; this change may
  remove only part of the measured overhead. G4 budget and G5 adoption stay open.
- Parent mapping is specific to the existing complete fixed-slot topology;
  arbitrary user tree topology is not changed or supported by this rewrite.

## Commits

- `df835e3`: preceding P7 evidence and negative value conclusion.

- Focused CPU regression: 80 passed. Production code and changed-test lint passed;
  staged change preserves the prior numerical oracle unchanged.

## Real CUDA correctness

- `81acf0b`: [T4 trainer artifact](../benchmarks/results/foundation/20260905T193001Z-cece3e8f/README.md),
  7 passed / 0 skipped. CPU/CUDA gradients, splits, leaves, tree predictions,
  bounded updates, Normal/Poisson ordinary/natural fits and rollback pass.
- Maximum adapter raw error 1.19e-7. Existing compact transfer counts unchanged.
  Performance remains to be measured; no speed claim from correctness runtime.

## Independent CPU package check

- Clean `a5bf26d`: [CPU installed-wheel artifact](../benchmarks/results/foundation/p7-fixed-slot-cpu-a5bf26d/README.md),
  7 passed plus public weighted demo and six exact plugin-free CPU predictions.
  Core wheel matches the GPU correctness/value bundles; plugins unchanged.
- Corrected prepare.py's descriptive profile metadata to say fifth host-profile
  and sixth memory-only fit. Worker behavior already separated them; the current
  running value bundle retains its original manifest verbatim. Its timed fits
  are unaffected, and its per-profile memory scope records the actual separation.

## Measured value and decision

- `a5bf26d`: [12-cell T4 matrix](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md),
  3 passed / 0 skipped. All default fits pass unchanged quality/fallback gates.
- Recorded default warm median 2.078694 -> 1.868019 s (10.13% lower); paired
  legacy ratio 13.899x -> 12.888x (7.27% lower). Legacy also got slightly faster,
  so do not attribute every raw percentage point to this rewrite. No statistical
  significance or general GPU advantage claimed from three splits.
- Keep this small optimization; it preserves all checks and original-row/tree
  parity. It addresses only part of the overhead. The original large regression
  remains: G4 still fails and the strict path does not replace legacy CUDA.
- Isolated seed-0 diagnostic tree/session 1.9099 s (builder 1.3641 s) versus
  objective .0866 s in a 2.0466 s profile. Remaining work is in tree/session
  validation/synchronization and cache traversal, not primarily objective math.
- CPU 80 focused tests, real T4 trainer 7 tests, independent wheel 7 tests with
  six exact plugin-free roundtrips, and value 3 tests passed. Lint and MkDocs
  passed. Exact peak/whole-process transfer trace and external adoption stay open.
- Final evidence audit: all uploaded file/wheel hashes, JUnit equality, unchanged
  measured implementation, and private-path scan passed. Recorded percentage
  reductions recomputed from raw medians/ratios. Final production/changed-file
  lint passed; docs build completed with existing griffe warnings.
