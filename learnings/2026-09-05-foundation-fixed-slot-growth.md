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
