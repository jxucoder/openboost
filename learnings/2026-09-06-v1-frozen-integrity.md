# 2026-09-06: Bind integrity to an evaluator-owned execution manifest

## Context

Sprint 070 must prevent producer omissions from shrinking required evidence.
The existing judge verifies a declared matrix and hashes but has no independent
expected-matrix input. Deleting a second required A1 fold from both sides and
rehashing leaves all A1–A13 present and passes declared integrity.

## Decision or Result

Extend the existing judge with an optional evaluator-owned frozen manifest. Require
exact canonical equality for the complete execution manifest, including provenance.
The CLI requires a separately pinned raw-file hash and a reference path outside
the producer output directory. Preserve the unanchored mode's explicitly limited
integrity semantics. Frozen match never implies a statistical or phase gate.

## Changes

- Judge/CLI frozen input and explicit match/hash reporting.
- Adversarial rehashed omission, provenance, backend, requiredness, duplicate and
  application changes; external file pin and path tests.
- Document trusted caller/reference assumptions and remaining isolation gap.

## Verification

58 focused artifact-judge tests pass. The initial omission regression failed on
the missing independent-input API while demonstrating old declared integrity passes.
Full CPU regression: **1010 passed**. Lint and docs pass. A nine-case synthetic
smoke accepts the valid case and rejects eight faults; clean-revision raw evidence
follows this commit. The smoke injects statuses, not real process resource failures.

## Failed Attempts

No attempt to relabel existing integrity as E-gate coverage. A path outside the
run directory does not prevent a same-user process from changing it. The pin must
come from an evaluator-controlled invocation; actual permission isolation remains open.

## Risks and Follow-ups

The frozen design itself must be generated/checked against all required R/C/A/E
obligations. Pinning a bad design does not make it complete. Actual resource and
label/verifier access isolation, source closure and full-search preflight remain
Sprint 070 work. Sprint 069 has no executable independent author runner or token
accounting yet; no author cost or compliance claim is justified.

## Commits

- Frozen execution-manifest binding; parent `e9058cc`.
