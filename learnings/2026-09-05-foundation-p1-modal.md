# 2026-09-05: Foundation P1 Modal Verification

## Context

The user requested continuation after P0. P1 must establish wheel-installed,
traceable GPU smoke tests and fail on missing/skipped tests or silent fallback.
Modal use was already authorized. Local CPU success is not GPU verification.

## Decision or Result

Use a dedicated `benchmarks/foundation/modal_app.py` instead of registering
foundation jobs on the legacy app: the legacy app registers source-mounted
jobs and broad dependencies, so reuse would build unrelated images and weaken
the installed-wheel boundary. Legacy entrypoints remain unchanged. The design
and execution checklist now reflect this implementation decision.

Pin the Linux amd64 CUDA 12.4.0 devel Ubuntu 22.04 image by registry digest;
export hash-locked CUDA/test dependencies from uv.lock, excluding unused JAX.
Use uv 0.12.1 for image installation, Python 3.12, a single T4, two CPU cores,
8 GiB requested memory, one container, retry=0 and timeout=300 seconds.

Prepare from a clean commit; verify all copied inputs and installed Python
package contents against the wheel. Disable automatic Modal source inclusion.
Two mandatory tests exercise CuPy/Numba pointer ownership with a real kernel
and the two-round Normal objective/native-tree call path. This is a smoke,
not complete CPU/CUDA parity, a held-out quality result, or a speed benchmark.

## Changes

- [Bundle preparer](../benchmarks/foundation/prepare.py): allowlisted upload,
  clean source SHA, wheel/data protocol and dependency hashes.
- [Offline validator](../benchmarks/foundation/runner.py): reject failures,
  timeouts, missing/duplicate/skipped cases, bad provenance and missing device
  execution evidence. CLI returns nonzero for invalid/missing reports.
- [Modal app](../benchmarks/foundation/modal_app.py): isolated image and bounded
  subprocess, environment/version collection and result persistence before
  validation. Image-build failures precede the local entrypoint and must be
  recorded separately.
- [GPU tests](../tests/foundation/test_smoke.py), isolated pytest config and
  conftest; [host contract tests](../tests/test_foundation_runner.py).
- Explicit result-format allowlist in benchmarks/results/.gitignore;
  [reproduction instructions](../benchmarks/foundation/README.md).

## Verification

Local commands use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.
The host contract test initially failed collection because the implementation
module did not exist; after implementation, all 12 tests passed. They exercise
failure propagation and completeness, not GPU execution.

Ruff checks/formatting cover the new support files. GPU execution and its
artifact will be recorded after committing this independently tested harness
and preparing a wheel from the resulting clean SHA.

## Failed Attempts

- Installed Modal 1.3.0.post1 has no `uv_pip_install_from_requirements` method;
  use its documented `uv_pip_install(requirements=[...])` API, confirmed by
  inspecting the installed SDK implementation.
- The first lint found an import ordering issue in the host test; corrected.

## Risks and Follow-ups

- CUDA smoke is pending at the harness commit. P2 still owns weighted Hessian,
  kernel fallback and full gradient/split/leaf/task parity checks.
- Function wall time excludes image startup/build and is not billed GPU time.
- The smoke's private imports/spies are test instrumentation; P6 external
  extension packages must use only the documented public API.

## Commits

- `f7eaf25` — P0 integration baseline.
- Harness code is committed before GPU evidence so the wheel has a clean SHA.
