# 2026-09-11: Curate a foundation checkpoint for public review

## Context

The user explicitly requested a PR after a scoped checkpoint was recommended.
The development branch contains several gigabytes of retained experiment
artifacts and unfinished follow-ups. Publishing that ancestry would obscure the
implementation review and distribute substantially more than the checkpoint.

## Decision or Result

Create a separate draft branch from public main. Preserve the full original
checkout and copy the verified checkpoint's source bytes without rewriting its
history. A draft PR is a review artifact, not engineering-v1 completion or a
successful validation of the curated package.

## Changes

- Source and self-contained tests: explicit checkpoint selection with a
  [hash and omission manifest](../planning/foundation-checkpoint-pr-manifest.json).
- Two derived comparison test files preserve 303 self-contained cases from mixed
  behavioral/archive modules with unchanged helper and test bodies.
- Documentation: current component boundaries and a compact historical audit
  report, with missing replay material and remaining acceptance gates explicit.
- CPU/docs workflows: defer automatic jobs while the PR is a draft; retain their
  commands and trigger them when it is marked ready. Skips are not passes.

## Verification

Verification results are recorded in the
[candidate metadata report](../planning/foundation-checkpoint-pr-verification.json).
They cover source identity, syntax/import closure, lint and local links only.
Numerical tests, installed consumers, CUDA, packaging and strict documentation
builds remain pending bounded Modal validation for this exact candidate.

## Failed Attempts

The existing archive-dependent suite cannot be assumed to replay on a branch
that omits its private execution history and new raw archives. The positive file
selection explicitly lists the omitted tests instead of treating them as passes.

## Risks and Follow-ups

Reconcile remaining omitted-test controls and historical replay dependencies before
marking the PR ready. All required application and evaluation gates remain in
scope, including the deferred, unpassed author study. The original development
branch remains the source of ongoing v1 construction and retained evidence.

## Commits

- This checkpoint commit is based on public main `31303e32` and copies production
  source from development checkpoint `91a519d`; it does not publish that ancestry.
