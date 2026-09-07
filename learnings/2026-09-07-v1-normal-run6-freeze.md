# 2026-09-07: Normal run-6 source and inference freeze

## Context

The user requested continued execution across local slices. `ac7b1e1` completed
external D2 construction after Normal operations, shared transactions and recipes.
All five GPU allowances remain consumed. Sprint 090-F requires a concrete package
before another upload or device request.

## Decision or Result

Freeze 67 files and 383 GPU cases, including all 212 run-5 verifiers unchanged,
170 Normal/D2 correctness cases and one separately labelled near-tie diagnostic.
Use the same bounded T4 runner with optional installed-extension and saved-artifact
accounting. The new guard tests require exact extension hashes/version, successful
CPU environment construction and all declared replay artifacts, in addition to
the existing exact test/core/version/source checks. Old evidence/verdicts remain
unchanged and their replay tests pass.

## Changes

- `cuda_aggregation_preflight.py`: optional external wheel installation, separate
  CPU environment build, installed extension source accounting and bounded JSON
  retention. Partial outputs from failed tests are retained; no retry is added.
- `normal_build_cpu_env.py`: build the exact core wheel, explicitly select the
  builder's Python for the build and fresh venv, install only NumPy and that wheel.
- Add a missing-input model replay in the fresh CPU environment. The D2 fixture's
  validation rows are all finite, so its eighteen trajectories alone could not
  establish the missing-input replay requirement. This makes 20 installed/fresh
  inference cases and 383 total; it changes no old hardware fixture or tolerance.
- `090-normal-run6.json` freezes the transitive test/reference imports, external
  project, kernels, runner, versions and exact output. `090-normal-run6-request.md`
  states the 67-file upload and one-run resource/acceptance boundary.

## Verification

All uv commands use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.

- `uv run --no-sync pytest tests/v1/test_cuda_normal_manifest.py
  tests/v1/test_cuda_symmetry_manifest.py tests/v1/test_cuda_resident_manifest.py
  tests/v1/test_cuda_splits_manifest.py tests/v1/test_cuda_aggregation_manifest.py
  -n 0 -q` — **49 passed**. Changed hashes, missing/consumed allowances, separate
  private-upload approval, reused output, extension mismatch, omitted artifact,
  failed CPU build and artifact byte/path violations are rejected.
- Full CPU regression with the existing `OPENBOOST_BACKEND=cpu` command:
  **1432 passed, one Linux-only skip**; `/tmp/openboost-090-f-cpu.log`.
- Ruff for production and changed support files, and `uv run --no-sync mkdocs
  build`, pass. Staged diff inspected before the local commit.
- Copy exactly `snapshot_paths` to `/tmp/openboost-090-f-snapshot`, then use
  installed Python 3.12 `-I -m pytest <declared files> --rootdir=<snapshot>
  --collect-only -o addopts= -q` from outside the repository: **383 collected**.
  Log `/tmp/openboost-090-f-snapshot-collection.log`. No CUDA package or device is
  needed for collection; collection is not hardware validation.
- `UV_OFFLINE=1 .../openboost-090-e-cpu312/bin/python` calling
  `normal_build_cpu_env.build('.', '/tmp/openboost-090-f-cpu-build-v4')` completes
  wheel build and fresh installation. The new venv's `python -I
  benchmarks/v1/normal_cpu_inference.py /tmp/openboost-090-e-replay` passes on a
  CPU-trained weighted/missing model without CUDA or D2 installed. Local builder
  dependencies use cached Hatchling 1.32.0; the frozen remote image retains 1.27.0.

## Failed Attempts

The first local no-isolation wheel build had no installed Hatchling; it failed
explicitly. The old pinned 1.27.0 was not cached locally. Install cached build
dependencies into the disposable verification venv, not the project environment.
Pin the build interpreter explicitly, matching the image's installed backend.
A subsequent local install attempted registry revalidation and failed on sandbox
DNS. `UV_OFFLINE=1` exercised the cached workflow successfully. No remote device,
upload, dependency download or permission escalation was used for those checks.

The new missing-input replay corrects a coverage gap found during package review:
existing D2 rows contain no NaN. It adds one distinct acceptance case before the
freeze, instead of claiming finite-row replay covers missing data.

## Risks and Follow-ups

Neither remote image construction nor new CUDA kernels/tests have executed.
Numerical/call-path failures remain possible and must be archived rather than
masked by changed tolerances. Metric/retention measurements are tiny correctness
diagnostics; they do not satisfy original P7, E4, quality/cost or author benefit.
The known near-tie remains an explicit structural-parity limitation.

The user now has a concrete upload/run decision. After that one result, consume
the allowance and reflect before any retry. No push or independent author attempt
is authorized by this preparation.

## Commits

- `ac7b1e1` — external D2 construction and original replay checks.
