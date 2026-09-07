# 2026-09-07: Public objective-owned loss-change operations

## Context

The user approved continuing after 092-A established the mathematical enclosure,
106-case numerical study and full 383-case historical mapping. The next boundary
is a reusable comparison operation, followed by three separate consumers. All
seven device allowances remain consumed.

## Decision or Result

`LossChange` is an objective-independent immutable record with validated bounds,
declared method/reason, derived status/estimate/uncertainty and strict threshold
comparison. `Normal.compare(problem, before, after)` now evaluates the actual
stored CPU snapshots without reading reporting metrics or geometry callbacks.
Both snapshots pass Normal row-domain checks before any unchanged/unresolved
shortcut, including zero-weight rows. Finite inputs beyond the bounded comparison
range return unresolved rather than inventing a result.

The private algebra factory shares a mathematical expression and accepts explicit
backend rounding operations. It imports no test/reference implementation. The
original-row Decimal expressions and 092-A interval prototype remain independent
test/evidence sources. CPU uses outward nextafter expansion under its documented
binary64 assumptions. CUDA construction follows as its own unverified slice.

## Changes

- Public `comparison.LossChange` and `Normal.compare`; unchanged loss, geometry,
  metrics, persistence and recipe decisions.
- Private bounded row algebra with explicit arithmetic dependencies; CPU domain
  validation and ordered weighted reduction.
- [092-B plan](../v1-sprints/092-public-comparison-operations.md), public Normal
  documentation and tests across the frozen study and validation/immutability cases.

## Verification

- First distinguishing tests failed with the absent `Normal.compare` operation.
  They cover the observed rate-four worsening candidate and `-2^-61` improvement.
- Focused tests evaluate all 106 declared cases against independent high-precision
  original-row differences, plus input/domain/record/threshold validation and
  metric/gradient callback independence. Results are recorded before each commit.
- CPU slice: **127 focused tests pass** and the full CPU suite passes **1633 tests
  with one Linux-only skip**. Production/changed support lint passes. MkDocs builds
  with its pre-existing run-6 evidence link warning.
- No real CUDA execution, emulator, independent author attempt or upload.

## Failed Attempts

- A new shape-error test initially accessed its removed second column before
  calling production. Move that fixture setup ahead of the shape mutation; the
  intended public validation then runs. No numerical contract or tolerance changed.

## Risks and Follow-ups

- The scalar arithmetic factory still needs device lowering/parity and cost
  evidence. A Python CPU pass does not verify its CUDA instantiation.
- CPU/device objective callbacks do not prove that arbitrary author-supplied
  bounds are mathematically correct. The bound owner remains explicit.
- Current backtracking, best-model selection and stopping are not fixed by this
  component alone. Keep distinct anchors and 065/068 ownership in 092-C; retain
  all historical outcomes and explicitly map revised consumers before hardware.

## Commits

- The verified public CPU slice is committed first; device construction follows.
