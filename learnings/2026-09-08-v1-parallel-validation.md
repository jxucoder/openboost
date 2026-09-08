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

## Slice B design

Use one 128-thread block per field, with strided row visits and an integer OR
barrier. Every lane participates, including lanes outside a short/tail row tile.
This follows the [Numba CUDA barrier contract](https://nvidia.github.io/numba-cuda/reference/kernel.html#numba.cuda.syncthreads_or).
It keeps the existing single launch and per-column flag allocation; no atomic
initialization kernel or floating reduction is needed. Domain errors and all
zero-weight rows remain checked. Only the field validation kernel and its launch
size change. Row-index validation and all algorithm arithmetic remain unchanged.

Thirty-nine new GPU cases collect without execution, covering independent flags,
tail rows, negative zero, invalid zero-weight rows and public failure recovery.
CPU tests cannot prove the barrier lowering or device speed; the next frozen
real-device run must verify both. Existing raw run-10 and run-8/9 archives remain
immutable; historical code is retrieved from its committed execution revision
when auditing after this production change.

Slice B local verification: full CPU regression passes 1975 tests with one
Linux-only skip in 14.98 seconds. Production/new-test Ruff passes and the new
39-case device collection succeeds. This verifies CPU regression and collection,
not CUDA correctness or acceleration. Slice A is committed at `680bf84`.

## Slice C result and reflection

Slice B is committed at `8db2aef`. The [run-11 request](../v1-sprints/105-validation-run11-request.md)
now freezes 88 files and 474 cases. The comparison verifies both installed cores,
exact stored input/model/prediction/quality evidence, all required repetitions,
cleanup and unchanged launch/flag exports before computing a warm ratio. Squared
CPU controls must complete and satisfy their frozen quality checks. Normal's
new case is explicitly original-versus-candidate GPU scope, not a new CPU pair.

The old full revised suite took 123.5 seconds, including 51.9 seconds for the
96-case compared runtime matrix. Repeating it with 485 seconds of child caps and
thirty seconds for baseline installation would be inconsistent with the planned
600-second test window. The frozen selection retains 431 existing cases (about
69 prior seconds), 39 new validation cases and four cost/profile cases, reserving
85 seconds for correctness/setup/audit. A hard deadline is still a cap, not a
promise that all work completes. One failure remains a failed invocation.

Actual local installation found that `uv venv --system-site-packages` sees the
base interpreter's packages, not the parent virtual environment's dependencies.
Merely adding the parent's directory also misses uv's resolved dependency paths.
The builder now appends resolved parent site directories after its own installed
core, then verifies every core file and declared dependency version. Its
[local check](../v1-sprints/105-baseline-install-local.json) verifies all 31 original
files and two saved CPU fits from a separate installation. The exact remote
Hatchling 1.27.0 is not cached locally; the offline check uses recorded 1.32.0 and
NumPy 2.3.5. All eighteen pinned Linux/CUDA versions remain a real-run requirement.

Full CPU regression: 1994 passed, one Linux-only skip, 12.95 seconds. The new local
judge/supervisor controls reject profile timings, actual child timeout, missing
repeats, source/model/metric mismatches, leaks and changed launch counts; a real
different CPU model fails the quality gate. Source/authorization guards and an
isolated 474-case collection pass. Separate instrumentation is never eligible for
fit ratios. All seventeen artifacts are bounded to 64 MiB; local lossless input
sizes total about 16 MB, leaving room for every repeated saved fit.
The final focused evidence/freeze/checkpoint suite passes 45 tests in 2.81 seconds.
Production and changed support/test Ruff checks pass. MkDocs builds successfully
with the pre-existing Sprint 090 evidence-link warning; no new warning is introduced.

This completes construction and a concrete reviewable packet, not hardware
acceptance. No new upload, remote invocation or speed claim has occurred.
Row-index validation is intentionally unchanged. The next result must decide
whether this small foundation optimization earns its complexity before further
optimization; it cannot substitute for required 080/081/082 or formal E4 scope.

## Run-11 authorization

The user explicitly approves the concrete 88-file Modal upload and one bounded
T4 invocation after construction commit `2101fd7`. Change only the protocol's two
authorization fields; preserve all 87 source hashes, case settings, deadlines
and zero-retry policy. Verify the local freeze guards, commit the approval state,
then execute once and audit the retained evidence before the planned retrospective.
The four local source/installation/authorization guards pass in 0.21 seconds.

## Real-device result

Run 11 executes once at clean `dd84247` and passes all 474 cases and three cost
gates. The [evidence archive](../benchmarks/v1/evidence/parallel-validation-105/README.md)
retains all seventeen JSON artifacts (33,985,980 bytes), twenty raw artifact
hashes plus the manifest binding, exact 88-source provenance, both installed
31-file cores and eighteen package versions. All 28 fits replay exactly on the
offline macOS host from retained input bytes, with independently verified scores.
The current allowance is consumed; the archived approved protocol stays immutable.

Warm original/candidate medians: squared 10,000 rows 2.838/2.749 seconds; squared
100,000 rows 13.513/8.947; Normal 10,000 rows 6.538/5.665. Original/candidate saved
models and predictions are exact across all repetitions. Large squared fit time
falls 33.79%, meeting the frozen 20% target. Squared CPU comparisons also complete
with relative task-score differences below 1.1e-7; Normal has no new CPU pair.
The separate validation operation's warm event interval falls 15.189 to 0.507 ms.
These include host enqueue/wait gaps and are not exclusive kernel measurements.

The first offline audit failed because it replaced slashes within parametrized
case names as well as module paths. Restricting normalization to the module path
matches the existing JUnit judge and preserves every original case identity. The
corrected audit verifies all 474 cases, twenty raw hashes and 28 saved fits. Five
local controls check numeric-summary tolerance, exact Boolean verdicts and
withholding complete medians from empty partial reports. No raw result changed.

The worker completes in 380.143 seconds and pytest in 377.754 seconds, within all
frozen caps. Total dispatch is 765.509 seconds including image construction/setup.
There are no timeouts, missing repetitions or retries. All 450 indexed files in
the prior 092/103/104 archives verify unchanged.
The final offline `analyze.py --check` reproduces the archived verdict and all
28 fit replays. All 45 focused evidence/freeze/checkpoint tests pass in 3.13 seconds.
Production and analysis Ruff checks pass; MkDocs builds with the unchanged old
Sprint 090 link warning. The result archive index binds 24 files; its own checksum
is intentionally not self-referential.

Keep the optimization and close this sprint at its planned reflection. Remaining
launch/synchronization counts do not identify another dominant kernel. Resume
required 080 recipe coverage next: objective operations plus persisted output
metadata, then 081 compatible batching and 082 real matched-quality cost. Public
composition, broad required applications and numerical/state correctness remain
the foundation goal; no external speed or adoption claim is established here.
