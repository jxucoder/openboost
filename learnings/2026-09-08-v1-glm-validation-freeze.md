# 2026-09-08: Bounded GLM hardware freeze

## Context

The user asks to continue foundation work. Sprint 107 closes its three local
construction slices at `703d9bd`, with 2,241 CPU passes and 153 GLM GPU cases
collected but unrun. All eleven hardware allowances are consumed.

## Decision or Result

Prepare the exact [run-12 request](../v1-sprints/108-glm-validation-request.md)
before seeking its separate upload/hardware allowance. Reuse the established
bounded launcher; add no performance campaign, agent study or production change.
Freeze 571 cases and 77 mandatory numerical/PTX/recipe artifacts. Preserve all
prior raw evidence and archived source revisions.

## Changes

- The new protocol enumerates 85 files, all non-protocol hashes, eighteen pinned
  packages, exact case IDs, resource/time limits and zero retries. Regeneration
  rejects approved/consumed or attempted packets; dispatch rejects pending
  allowances, changed sources, reused output or a different destination.
- GLM evidence uses a subdirectory of the existing collector root. Actual PTX
  is losslessly JSON-encoded rather than discarded as an uncollected `.ptx` file.
  Numeric/PTX result assertions follow report writes. Sixteen recipe reports add
  actual input dtype/shape/base64 bytes and use the fresh CPU-only interpreter.
- The isolated wheel/snapshot check verifies complete import closure and all
  571 IDs without executing CUDA. Existing regression assertions remain intact.

## Verification

Forty-five focused freeze/manifest/retention checks pass in 0.51 seconds, including
25 new packet controls. Installed collection matches all 35 production Python
files and all 571 test IDs; the complete payload has 85 files and 1,443,454 bytes
in the pending freeze. Fixture snapshots round-trip exact bytes, including NaN
bit patterns. Missing/duplicate/skipped/failed cases, missing artifacts, source or
package mismatches, failed CPU setup and timeouts reject a synthetic complete
verdict. These are harness controls, not device correctness evidence.
Production and changed-file Ruff pass. The MkDocs build passes, with the existing
historical Normal evidence-link warning unchanged. The isolated collection also
builds the current wheel offline before checking its source identity.

## Failed Attempts

The first lint pass identified import ordering in the new freeze utility; it was
corrected before the frozen collection. No CUDA dispatch or upload was attempted.

## Risks and Follow-ups

New kernel compilation and complete recipe behavior on a real T4 remain unknown.
The 600-second shared test deadline may be insufficient; a timeout remains a
failure with no retry. Image building lies outside the 900-second function cap,
so the request makes no guaranteed dollar-cost claim. Await only this concrete
allowance, execute once, retain raw results and reflect before new construction.
The pending freeze changes no required application scope or formal gate status.

## Commits

- `95b5232`: independent convex comparison mathematics and counterexamples.
- `67788ba`: bounded resident binary/Poisson comparisons.
- `703d9bd`: compared scalar recipes and independent trajectory controls.

## Run-12 authorization

The user's subsequent "continue", followed by explicit "approve", authorizes
the exact 85-file Modal upload and one bounded T4 invocation described in the
request. Only the two protocol authorization fields change; all 84 frozen
non-protocol source hashes remain unchanged. The 25 focused freeze checks pass
in 0.30 seconds before recording approval. Commit the authorization before
dispatch so the manifest records a clean execution revision. No retry or extra
invocation is authorized; preserve the result and stop after its retrospective.
