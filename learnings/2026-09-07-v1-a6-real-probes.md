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
