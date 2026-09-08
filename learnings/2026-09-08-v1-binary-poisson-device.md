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

## Slice C result and reflection

`bed9605` commits the resident component construction. The next class-schema test
fails first because DeviceRun reaches its device requirement without rejecting
different labels. Validation now precedes device work; exported Model records keep
the immutable training schema in the existing serialization format.

The two-round independent oracle enumerates original-row splits and sums in
float64, with explicit float32 state/geometry boundaries. An initial count fixture
had equal-gain root splits around a zero-weight row. Positive weights remove that
ambiguity from composition checks; the earlier domain cohort keeps zero weights.
Poisson's intercept can make a depth-zero step stationary, so required gradient
change is tested on nontrivial trees. Both are local fixture corrections before
any device run, not tolerance adjustments in response to hardware.

The six new device composition cases check numeric/missing routing, leaves, two
rounds, different validation order/offset/weights/exposure, task metrics, immutable
prior states, explicit rejection, stream restoration, cleanup and saved CPU-only
inference. Forty-four total GLM GPU cases collect, without execution. Independent
oracle/configuration/comparison/archive tests pass: 85 in 8.28 seconds. Ruff passes;
documentation builds with its existing historical link warning. Wheel/sdist build
passes offline with normal build isolation, and the wheel's new modules/runtime
match source bytes. The initial no-isolation attempt lacked Hatchling; no dependency
or environment mutation was necessary.

Full regression using `OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/ -m
"not gpu and not benchmark" --tb=short -n 0` passes 2,065 tests with one Linux-only
skip and 771 deselected cases in 75.18 seconds. Environment remains macOS 26.3,
Python 3.12.12 and NumPy 2.3.5, with `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.
The skip and deselections are not device evidence. Consumed packet checks read
their frozen execution revision; no old archive or freeze file changed.

This third construction slice supports the foundation's abstraction boundaries,
not a device quality/speed or adoption claim. Automatic decisions still require
objective comparison. Sprint 107 records exact next construction/acceptance order:
independent sign/bound counterexamples, resident comparison, full recipe consumers,
then a bounded hardware request. No extra GPU/model call, upload, push or external
evaluation occurred. Preserve all R/C/A scope and existing evidence.
