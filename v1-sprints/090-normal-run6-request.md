# Run 6: Normal and installed D2 correctness package

Status: executed at `4143d18`; 381 passed and two failed. Allowance consumed;
no retry authorized. See the [result](../benchmarks/v1/evidence/cuda-normal-090/README.md).
Exact package: [090-normal-run6.json](090-normal-run6.json).
Construction baseline: `ac7b1e1`; the protocol freezes every uploaded file except
itself by SHA256. Dispatch requires a clean checkout and records its actual SHA,
the complete protocol and all 67 source hashes, including the protocol's own hash.

## Concrete request

Upload the **67 explicitly listed repository files to Modal** and execute **one
T4 invocation**, with **2 requested CPUs, 8192 MiB requested container memory,
900 seconds maximum function duration and 600 seconds maximum pytest duration**.
There are no retries or parallel containers. Both allowances cover this package
only; all five previous invocations are consumed. CPU image construction installs
the pinned dependencies, the exact OpenBoost snapshot and external D2 package;
it also builds a separate CPU inference environment from the same source wheel.

The upload contains 28 production Python files, the declared verifiers and their
reference import closure, the three-file external cohort project, build/replay
scripts, package metadata and this run's JSON protocol. It contains no sealed task
cards, credentials, Git directory, complete workspace or independent-author data.
The exact list and per-file hashes are in `frozen_sources`; the JSON protocol adds
the 67th file. No external dataset is needed; fixtures are committed synthetic
original-row cases. There is no push, release, publication or leaderboard action.

## Acceptance and retained evidence

- All **383** preregistered cases must appear exactly once without skips/failures.
  The first 212 preserve run 5's test files, order and tolerances unchanged.
- 170 new correctness checks cover Normal geometry, mapped transactions,
  joint/ordered recipes, installed D2, numerical retries, ownership and saved
  CPU inference. One additional near-tie diagnostic validates its recorded sums,
  scores and prediction differences; it does not repair structural parity.
- Installed core and extension source hashes, pinned package versions and uploaded
  snapshot hashes must agree. The fresh CPU environment has NumPy and OpenBoost,
  with no CUDA dependencies or training extension; missing inputs have a dedicated
  model replay in addition to eighteen D2 trajectories.
- Retain pytest output, JUnit, manifest, verdict and **76 declared JSON artifacts**
  for nineteen saved models, their inputs, measurements and fresh CPU replays.
  Additional JSON is capped at 2 MiB; partial files from failed tests are retained.
- Report first/repeated fit times including context/preparation, independent-field
  uploads, all trials and model export. Report CPU model prediction separately.
  Record synchronization/transfer/copy counters and logical/pooled retention.
  CPU fixture creation and supplied fitted binning precede the timer; device
  preparation is timed. Earlier tests may compile kernels; first-in-case timing is not process-cold or
  matched-quality performance evidence. Existing aggregate fixtures remain bounded
  at 8192 rows, 32 features and 32 bins; tiny training fixtures remain at no more
  than 8 rows, depth 2 and 24 rounds, with a 16 MiB private device allocation pool.

Original P7 reproduction, E4, independent-author accounting/isolation, remaining
required CUDA recipes and the full R/C/A scope remain open. No quality, speed,
adoption or formal phase-exit claim follows from this run.

## Dispatch and stop boundary

The following is the approved dispatch procedure retained for provenance. This
allowance is now consumed and must not be reactivated to rerun the command.

After explicit approval, set this protocol's compute and upload authorization
fields to `approved`, commit, recheck the source freeze and run:

```sh
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.cuda_normal_preflight benchmarks/v1/evidence/cuda-normal-090
```

The entry point rejects pending/consumed authorization, a dirty checkout, changed
sources, another output location or an existing output directory before importing
Modal. Archive the result, mark the allowance consumed, reflect on actual failures
and stop at the planned retrospective. A failed run does not authorize a retry.

## Local verification

All **1432 CPU tests pass**, with one Linux-only skip; 49 dispatch/manifest checks,
Ruff and documentation build pass. The exact copied 67-file snapshot collects all
383 GPU cases outside the repository using an installed wheel and isolated Python;
none of those cases executes locally. The CPU environment builder/replay workflow
passes locally using the cached Hatchling 1.32.0. The remote package retains the
prior pinned Hatchling 1.27.0; its image build and all CUDA behavior remain unrun.
See the [learning record](../learnings/2026-09-07-v1-normal-run6-freeze.md).
