# 2026-09-05: P3 CPU extension contract

## Context

P2 baseline passed at 99621ae; P3 introduces a small CPU extension facade while
retaining the existing trainer loop. The user requested continuation.

## Decision or Result

Implement in three verified slices: objective/arrays, builder/schedule, then
persistence/early-stop consistency. CPU-only execution is explicit; requested
CUDA either fails preflight or warns and selects the entire CPU path.

## Changes

- `experimental.Booster`, immutable `ExecutionContext`, shared `TrainerConfig`
  and a distribution adapter. The facade invokes the existing trainer.
- Validate complete channel keys, contiguous float32 statistics, nonnegative
  curvature, finite data/weights, positive total weight and explicit devices.
  Inputs are read-only views; aliased outputs fail without per-round host copies.
- Add min_gain to the single trainer config and propagate it to tree builders;
  a supplied per-fit generator can be shared with extension contexts.

## Verification

- Initial independent two-channel objective test failed import before creation.
- The first implementation exposed a shared weight-validator boundary: it
  allows all-zero weights and Inf. The experimental facade now rejects them
  before binning without changing legacy validation semantics.
- Slice 1: 53 passed (23 experimental, plus existing foundation/formula/survival
  tests); production/new-test lint passed. No experimental GPU execution is
  claimed by P3.

## Failed Attempts

- All-zero weights initially passed; fixed at experimental preflight.

## Risks and Follow-ups

- Builder/schedule dispatch and plugin-free inference persistence follow in P3.
- Read-only views prevent accidental mutation; this is not a sandbox against
  deliberately hostile Python plugins accessing underlying memory.

## Commits

- `4c16204` — completed P2 evidence.
