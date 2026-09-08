# 2026-09-08: Preserve performance evidence and parallelize field validation

## Context

The user approves Sprint 105 after run 10 exposes input-regeneration differences,
lost final artifacts after later timeouts, and costly serial CUDA validation.

## Decision or Result

Build exact input snapshots and recoverable per-fit evidence before changing the
validation kernel. Keep run 10's sources and raw archive immutable. Its historical
implementation remains available at `c8f7ebc`. New performance evidence must use
the same stored inputs in both arms and complete declared repetitions.

## Changes

The approved order is recorded in
[Sprint 105](../v1-sprints/105-parallel-validation-and-reproducible-cost.md).

## Verification

First failures to cover: changed targets with unchanged predictions, byte/identity
mismatch on input loading, and interruption after a fully recorded first fit.
CUDA validation must check invalid tail rows and all public error/cleanup paths
on real hardware; local collection alone cannot verify kernels.

## Failed Attempts

Run 10's timeouts and nonportable generated identities remain in its archive.

## Risks and Follow-ups

Parallel validation is a candidate optimization, not a proven dominant-kernel fix.
Preserve all zero-weight-row checks, numerical reduction order, ownership and
accepted/best-state decisions. Freeze exact sources and realistic time budgets
before requesting the next hardware allowance. Author/model studies stay deferred.

## Commits

- `840cc41` records the preceding performance checkpoint and proposed response.

## Slice A result

Exact numeric inputs now include lossless little-endian array bytes, checksums,
Problem identities and a required snapshot binding. A target change cannot reuse
the old binding even when model predictions remain unchanged. Every completed
fit saves replayable model/predictions/quality before another fit can time out.
Timeout/running/error records never acquire complete-case eligibility from their
partial fits. The original run-10 benchmark and archive remain unchanged.

Ten new and sixteen existing focused CPU checks pass (26 total, 1.24 s), including
real process termination during the second fit and replay of the retained first
fit. The [isolated check](../v1-sprints/105-input-replay-local/verification.json)
replays two Normal fits with data generation disabled and only the installed core
and NumPy present. It records 34 production/support source hashes, environment,
inputs, models and exact commands; this is local format verification, not cost
or GPU evidence. Production and support lint pass.

The first local installation attempt used `--no-build-isolation`, but Hatchling
was absent from the project runtime. Use the declared build environment through
`uv build --offline` instead; the actual replay environment still contains only
NumPy/core. The final replay reruns after support metadata/validation is complete.
No project environment mutation or network installation is needed.
