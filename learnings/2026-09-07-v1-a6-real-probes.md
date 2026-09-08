# 2026-09-07: Execute the first real A6 resource probes

## Context

The full OpenBoost submatrix is compiled but no full-round real worker is qualified.

## Decision or Result

Use the existing frozen exporter and only upload fold-zero train/validation input.
Run the two named 300-round configurations sequentially with unchanged patience,
resource policy and no retries. Stop on any fit or replay failure.

## Changes

- A narrow Modal harness validates the compiled plan and input freeze hashes.
- Privilege-dropped fitting records peak RSS; fresh inference must match exactly.
- Each invocation uses a separate output root, reclaimed before a subsequent job.

## Verification

All five local folds export successfully against the committed freezes. Fold-zero
packet hash is 0750282946e2e2e5636117db6564652c504b1958ff363c20474816dd3a849f53.
Source lint passes before remote dispatch. Actual results follow the clean commit.

## Failed Attempts

No resource outcome yet; a successful packet export does not qualify model fitting.

## Risks and Follow-ups

Two configurations on one fold cannot qualify all 160 fits or comparator methods.
Address space and guest RSS remain different quantities; container caps are requested.
No real test material is uploaded or scored. Fresh inference is evaluator-owned.

## Commits

- Real A6 probes; parent `c9201ae`.

### Dispatch blocked before upload

Automatic approval review rejected the Modal command before execution. It requires
explicit authorization for this real Parkinsons train/validation payload despite
prior general Modal authorization. No remote probe or upload occurred. Do not
retry via another mechanism. The local packet is 1494662 bytes: training features
3487 by 38 and targets 3487 by 2; validation features 1151 by 38 and targets 1151
by 2, plus 1151 validation row IDs. No test files are allowlisted.

Local CPU validation: 1038 passed, one Linux-only skip; lint, compilation and docs
pass. The committed harness is ready for review; actual resource/replay outcomes
remain unknown until payload upload is explicitly approved.

### Approved dispatch results and reflection

After explicit payload approval, the two real probes run successfully from clean
`9387d9f`. [Raw evidence](../benchmarks/v1/evidence/a6-real-probes-070/README.md):
shared trees take 1077.14 seconds with 59 completed rounds and 108097536-byte peak
RSS; independent trees take 767.13 seconds with 65 rounds and 136032256-byte peak
RSS. Both stop by frozen patience 50 within their 300-round budgets and replay
validation predictions exactly in a fresh process. No test data is uploaded.
All 31 source hashes, 20 artifacts, packet/plan hashes and target scales verify.

These are two scoped resource passes, not full-search qualification. Practical
runtime is now directly measured and substantial despite early stopping. Before
expanding to the remaining configurations, inspect a bounded profile of this exact
input and configuration, then decide whether a targeted change is justified. No
bottleneck or comparative speed claim follows from wall time alone. Keep the
complete ledger, comparator matrix and author accounting open. This is a reflection
checkpoint; no additional jobs or optimization are included in this slice.
