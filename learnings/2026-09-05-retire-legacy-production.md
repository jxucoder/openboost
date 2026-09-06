# 2026-09-05: Retire the old production implementation for a clean v1 rebuild

## Context

During Sprint 001 the user asked about deleting everything and clarified the
scope as old production code, rebuilding v1. This overrides the earlier option
to keep old sources until later F1/F5 cleanup.

## Decision or Result

Retire the entire old package implementation at a single Git boundary. Keep an
explicitly unfinished v1 namespace, independent references, historical mathematical
tests and raw experiment evidence. Historical reproduction uses `50acfc6`.
This is a clean implementation start, not a completed foundation or feature release.

## Changes

- [Sprint 002](../v1-sprints/002-retire-legacy-production.md) is the authoritative
  execution/verification/reflection record, including the user-directed sequence change.
- Remove old models, trainer, CPU/CUDA backends, core, experimental and distributed
  modules. Remove stale bytecode/JIT caches; no compatibility layer remains.
- Rebuild default test/docs/CI entry points around current v1 work; retained
  historical tests are excluded, not relabeled as passing/skipped coverage.
- Set package metadata to `1.0.0.dev0`, drop retired runtime/autodiff/distributed
  dependency surfaces and refresh the lock. No package is published.
- README and current docs explicitly state there is no training API. Old GPU and
  publishing jobs are unavailable until their v1 gates have real implementations.

## Verification

- 55 reference tests pass through the default root test command, without skips;
  changed production/test support lint, lock check, strict docs and offline
  sdist/wheel builds pass. Isolated wheel import and archive inspection confirm
  no old modules remain. Full commands and boundaries are in Sprint 002.
- Four local workflow schemas, 14 Markdown files/76 local links and whitespace
  checks pass. No benchmark artifact or committed reference was changed by cleanup.
- Remote CI, CUDA and production model gates were not run; the namespace has no
  training API. Historical tests excluded from discovery are not v1 passes.

## Failed Attempts

- Offline lock regeneration could not resolve uncached cross-Python metadata.
  A normal registry-backed uv lock was needed; this is an environment limitation,
  not an excuse to hand-edit dependency hashes or leave a stale lock.

## Risks and Follow-ups

- The checkout intentionally cannot train until new public components arrive.
  Historical commands require the recorded revision. New component tests must
  recover relevant correctness cases without preserving obsolete API contracts.
- Return to remaining F0.2 references, then F0.3 and F1. All application and
  quantitative gates remain required; cleanup is not evidence of correctness or speed.

## Commits

- `50acfc6` — old production implementation plus independent v1 references.
- This slice: `refactor: retire legacy production code for the v1 rebuild`.
