# 2026-09-06: Evaluation-side local held-out seal

## Context

Sprint 016 requires H1/H2 cards and mathematical/state verifiers outside the
foundation designer's context. The user authorized a separate evaluator agent.

## Decision or Result

An evaluator agent authored and locally sealed the package. The public
[manifest](../benchmarks/v1/heldout-manifest.json) contains only hashes, storage
paths, validation status and independence/storage limitations. No task or verifier
contents were sent to the foundation designer. This is a preparation artifact,
not an author-evaluation result or closure of F0.3.

## Changes

- Public manifest: exact package and member SHA-256 values, local custody path,
  validation status, pending execution gates and independence limitations.
- Private package: locally frozen evaluation material; do not inspect it during
  interface design. Preserve the private bytes before temporary-directory cleanup.

## Verification

- Internal mathematical/state verifier self-validation: passed.
- Diagnostic candidate entry point and adversarial output rejection: passed.
- Invalid-parameter rejection: passed.
- Ruff check of both private Python support files: passed.
- Archive member hashes and public manifest consistency: verified locally.
- Exact commands and environment are retained inside the sealed package. This
  record deliberately omits content-bearing details.

## Failed Attempts

- Initial uv invocation could not access its default sandboxed cache. Repeating
  with a dedicated temporary cache succeeded; no environment packages changed.

## Risks and Follow-ups

- Independence is a separate agent context on the same system and filesystem,
  not an independent human, independently selected model cohort, or OS security
  boundary. Public development material was visible to the evaluator.
- Read-only file modes prevent accidental edits, not access by the parent or
  future candidates. The archive has been copied opaquely and hash-verified into ignored
  `build/v1-heldout/package.tar`; this avoids temporary cleanup but still requires
  private preservation before build cleanup. It is not backed up.
- Real library integration, actual fresh-process persistence, comparator and
  cohort freeze, restricted execution, and E5 attempts remain pending. Numerical
  diagnostic success does not satisfy these gates.
- If task contents inform foundation design, retire the affected held-out and
  replace it before evaluating unseen-task performance.

## Commits

- This slice: `eval: seal independently authored held-out package metadata`.
