# Sprint 070: Coverage ledger and trustworthy selection judging

Status: in progress; evaluator-freeze integrity slice implemented, broader gates open. Mapping: N4 / B02–B11 / C6 / F0.3 and CPU F1 entry/exit auditing.
Entry: inventory/source work can start now; practical execution preflight follows 068.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

## Outcome and first failing checks

Required evidence cannot disappear from a report, and candidates cannot access
sealed test labels or escape frozen resource caps. First feed the existing judges
missing/duplicate/stale required cells and an omitted application. Separately run
fault workers that attempt to read test labels, exceed RAM/time, produce NaNs,
claim a wrong backend, or exit unsuccessfully. Record which failures are already
handled and which demonstrate a real gap.

## Work

- Join every R1–R9/C1–C7/A1–A13 and E0–E7 obligation to revision, verifier, raw
  artifact, environment, status and next action. Derive the expected set from the
  frozen protocol; producer omission cannot shrink it. Manual rows are navigation.
- Extend existing integrity, search and quality tools with independent gate checks;
  do not create another universal runner. Distinguish integrity_pass from actual
  quality, authoring or device acceptance. Missing evidence must not pass a gate.
- Enforce test-label separation at the process/container/access boundary and RAM,
  time and thread caps. A selection receipt or hash alone is not access isolation.
  Confirm validation-only selection precedes any authorized test release.
- Start A4 MSLR terms/source closure and A10 Veteran provenance closure. Freeze an
  equivalent documented substitute before evaluation if necessary; keep unresolved
  rows open and progress unrelated sources independently.
- Audit actual F0/F1 prerequisites against public CPU call paths and installed
  workflows. List each remaining blocker to 077; do not infer exit from recipe count.

## Acceptance and reflection

Omission, duplication, stale identities, invalid outputs, worker/resource failures,
and attempted label access cannot produce a pass. A small valid search selects
using validation, releases only the selected model and reproduces its independent
score. This synthetic harness test does not pass real A13 or E3.

Publish the complete ledger and enforced-environment manifest, plus the frozen
job-count/resource preflight for [071](071-real-multioutput-selection.md). Keep
source issues and formal phase blockers explicit. If a missing judging component
needs a substantial implementation, split a focused follow-up before expensive jobs.
Reflect on whether the report now proves conditions or merely inventories files.

## Results

Not run. Existing integrity and short quality reports do not establish these gates.

### Plan and first counterexample

First inspect existing coverage/integrity tools, reproduce a producer-shrunk matrix,
add an evaluator-owned execution freeze to the existing judge, verify malformed/
rehashed cases and CLI pin checks, then commit a reproducible smoke. Separately
inventory remaining accounting/isolation and complete-coverage work before large jobs.

The old judge correctly rejects missing records relative to its manifest, but a
producer can delete a second required A1 fold from both manifest and records and
recompute all cache keys. All A1–A13 are still present and declared integrity passes.
The new optional evaluator freeze rejects that attack, changes to provenance/
protocol/backend/requiredness and duplicate/omitted cases even after rehashing.
The CLI requires a pinned reference-file hash outside the producer output root.
58 focused tests pass, including the deliberately failing initial counterexample.

This is an execution-manifest binding, not the complete R/C/A/E coverage ledger.
The reference and invocation must be evaluator-owned; filesystem/process isolation
and actual experiment completeness remain open. No E-gate result is emitted.

Full regression: **1010 passed**; lint/docs pass. A nine-case reproducible smoke
includes the valid matrix, rehashed fold omission/code change, missing/duplicate
records, wrong backend, worker-error/timeout status and nonfinite metrics. These
are injected judge inputs, not evidence of actual OS resource/access enforcement.

### Real worker retention follow-through

The current A1–A12 worker still inherited full traces after 068. Three failing
checks observed full payloads through actual squared, Normal and A5 quantile
calls. It now explicitly passes summary retention and records that policy in
training metadata, without adding a search parameter. All current-worker direct
recipe and fresh-prediction parity checks run before commit. Other worker families,
including composed frequency/severity, remain a separate preflight audit item.
This fixes a known diagnostic-memory policy gap; it is not a full-search launch.

### Evidence checkpoint

The [clean synthetic smoke](../benchmarks/v1/evidence/frozen-judge-070/README.md)
accepts one valid bundle and rejects eight faults. Source hashes match `63700db`.
Worker summary policy at `1a7bfd5` passes 86 adapter tests and 1013 total CPU tests,
including direct full-recipe and fresh inference checks. No core production API
changed in these slices.

Reflection: two concrete gaps are closed, but their evidence must not be promoted
to full isolation or coverage acceptance. A producer cannot shrink an externally
frozen execution matrix; the evaluator must still construct that matrix correctly
and protect it. Summary retention is explicitly selected in the current worker;
the full-search environment is still unqualified. See the
[readiness inventory](070-readiness-inventory.md) for the next bounded work.
Sprints 069/070 remain open; no independent author or expensive search was launched.
