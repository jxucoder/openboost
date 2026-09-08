# 2026-09-08: Binary and Poisson objective boundaries

## Context

After the verified field-validation optimization, the user asks to continue.
Sprint 080 requires more applications through the same foundation. Current device
objective callbacks support new geometry, but DeviceRun export drops class metadata.

## Decision or Result

[Sprint 106](../v1-sprints/106-binary-poisson-device-components.md) builds independent
numerics, resident objective components and class-aware two-round composition.
Full recipe acceptance/best/stopping decisions remain an explicit next slice.

## Changes

Freeze the required cell inventory, stored-input domains and first counterexamples
before new kernels. Bind consumed run checks to committed sources rather than
requiring future production to remain identical to the old packet.

## Verification

First failures: missing/skip/fallback evidence, stable signed-margin tails, count
rounding and exposure/weights applied twice. Real-device checks must also validate
split/leaf/state/prediction/metric behavior, not only geometry shapes.

## Failed Attempts

None in this slice yet. Run 105's immutable audit remains the prior cost evidence.

## Risks and Follow-ups

Float32 representability is a declared device restriction. Do not treat CPU
oracles, collection or prescribed updates as CUDA acceptance or stable recipe
decisions. No new hardware allowance, external evaluation or author call is included.

## Commits

- `346dee5`: preceding real-device optimization evidence and retrospective.

## Slice A result

The independent stored-input oracle uses existing float64 likelihood formulas and
Decimal Poisson initialization, with representation checks separate from equations.
Thirty-eight mathematical/scope controls pass, including finite differences,
tiny nonzero binary derivatives, offset-800 initialization, target aliasing and
exactly-once weights/exposure. The required scope includes fourteen recipe/batching
cells; removing any cell or replacing full checks with geometry alone fails.

Five archive/freeze tests also pass. Consumed run-11 hashes are checked against
`dd84247` bytes through Git; a negative test forbids reading the current production
tree for those consumed checks. All frozen files and raw results remain unchanged.
Total 43 tests pass in 7.83 seconds; changed-file Ruff passes. This is local oracle
and accounting validation, not real-device correctness or full recipe acceptance.

## Slice B result

At `270fe4b`, the independent contracts were committed before kernels. The new
`device_glm` module composes the existing workspace/ownership and field operations.
Family-bound prepared records prevent a binary target from silently entering the
squared operation merely because both have one raw column. Poisson stores exposure
separately and rejects counts changed by float32 conversion. Loss and geometry
share the same per-row domain, including weight-zero invalid rows.

The kernels use stable complementary binary probabilities, signed-margin loss,
float64 log-sum-exp initialization and Poisson log-factorial. Numba's official
[CUDA Python math support](https://nvidia.github.io/numba-cuda/user/cudapysupported.html)
lists `log1p` and `lgamma`; actual compilation remains a real-device gate.

Twenty-five new host tests and the 43 prior oracle/freeze checks pass (68 total,
9.80 seconds). All 38 new CUDA cases collect; none was executed or counted as a
pass. Ruff passes across production and changed tests. Review caught an incorrect
test fixture keyword (`max_bins` instead of `bins`) before any device submission.
The public execution document also replaces its stale run-11 pending statement
with the already committed bounded measurements, preserving their synthetic scope.

Next: class-aware export, prescribed two-round composition, independent split/leaf
and task-metric checks. Recipe acceptance/best/stopping integration remains later;
the absence of a comparison callback fails explicitly when objective comparison
is requested. All old frozen source files and raw archives remain untouched.
