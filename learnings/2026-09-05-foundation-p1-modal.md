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

`uv run --no-sync ruff check src/openboost/ benchmarks/foundation
tests/foundation tests/test_foundation_runner.py` passed, as did staged
whitespace checks. The wheel was built from clean harness commit `3101a44`.

The authorized single-T4 smoke completed successfully: **2 passed**, no skips,
Modal CLI exit 0. Offline revalidation with `python -m
benchmarks.foundation.runner` also passed. Raw evidence and reproduction:
[20260905T080803Z-c5ced00e](../benchmarks/results/foundation/20260905T080803Z-c5ced00e/README.md).

Verified 37 installed Python files against the wheel, device pointer/lifetime
interop, 2 device objective calls and 4 native tree calls. Remote function wall
time was 14.66 seconds (not billed GPU duration); pytest was 9.16 seconds.
The T4 driver was 580.95.05. CuPy reported CUDA runtime 12.9 despite the pinned
12.4 toolkit base; record runtime, toolkit and driver separately. Small-grid
under-utilization warnings are expected for this smoke.

## Failed Attempts

- Installed Modal 1.3.0.post1 has no `uv_pip_install_from_requirements` method;
  use its documented `uv_pip_install(requirements=[...])` API, confirmed by
  inspecting the installed SDK implementation.
- The first lint found an import ordering issue in the host test; corrected.
- Sandboxed wheel preparation could not resolve PyPI for hatchling. The same
  clean-source preparation succeeded with the authorized network permission.

## Risks and Follow-ups

- CUDA smoke passed for the harness commit. P2 still owns weighted Hessian,
  kernel fallback and full gradient/split/leaf/task parity checks.
- Function wall time excludes image startup/build and is not billed GPU time.
- The container reports architecture, visible CPU count and host RAM alongside
  requested limits. Exact physical CPU model was not collected; add it to P2
  performance provenance when available. No speed claim is made here.
- The smoke's private imports/spies are test instrumentation; P6 external
  extension packages must use only the documented public API.

## Commits

- `f7eaf25` — P0 integration baseline.
- `3101a44` — harness implementation, independently verified before GPU use.
- Raw GPU evidence is committed separately from its source to avoid circular
  source hashes.
