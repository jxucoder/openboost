# 2026-09-07: Author material and device ownership prerequisites

## Context

The approved 085 work starts with D1/D2 authoring preparation and the actual CUDA
ownership seam, instead of further CPU resource probes.

## Decision or Result

The CPU call path forces NumPy/bytes storage at data, fields, topology and raw
state; per-candidate Python callbacks also prevent resident execution. Kernel
substitution alone is insufficient. Record the explicit ownership/export and
batch-operation boundaries before implementing device storage. No private trainer
bypass or silent fallback is acceptable.

D1/D2 author materials must not ship existing solved extensions. The preparation
command creates a narrow author directory with selected current docs and a wheel;
it pins evaluator inputs separately without copying them into that directory.
The wheel exposes legitimate public implementation for inspection. No agent runs.

## Changes

- 069-author-packet contains task cards and budget/delivery instructions.
- prepare_author_packet builds only from a clean tree and records missing actual
  accounting/isolation, model/runner, incumbent audit and verifier invocation.
- 078-device-boundary-audit identifies current call paths and the next device
  storage/fields/histogram slice with ownership acceptance requirements.

## Verification

Before building, verify the dirty-tree guard rejects output creation. Then commit
and build from that clean revision, inspect wheel members and exported file hashes,
and verify no solutions/evaluator files are copied into the author directory.
Documentation/lint and packet results are recorded below.

## Failed Attempts

No independent authoring or device attempt. Historical docs outside docs/v1 refer
to retired APIs by design; use only current docs/v1 in the author-view allowlist.

## Risks and Follow-ups

The evaluator files do not yet constitute a standalone dispatch/verifier. Selected
docs retain ordinary relative links, some pointing outside this deliberately small
view; they do not grant filesystem or network access. Validate the final author
instructions and navigation when the complete attempt environment is frozen.
Actual token/time observation, independent arm audit and OS isolation remain open.
The CUDA API and dependencies remain unimplemented/unpinned. Next implement the
execution-owned storage seam and close the accounting protocol in bounded slices.

## Commits

Preparation and source audit first; clean packet evidence follows separately.

Preparation validation: the dirty-tree guard rejects before creating output.
Changed-file lint, MkDocs build and whitespace checks pass. Existing CPU tests
were not rerun for this packaging/documentation-only slice.
