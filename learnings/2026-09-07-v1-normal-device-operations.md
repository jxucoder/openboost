# 2026-09-07: Resident Normal operations before runtime integration

## Context

The numerical freeze `f75f501` precedes kernels. The user asked to continue across
verified slices without stopping. 090-B constructs the first Normal device
components while preserving the existing scalar path and all 212 old verifiers.

## Decision or Result

Normal preparation/base/geometry/loss are explicit public operations. Generic
diagonal direction and per-column least-squares fields can serve other algorithms.
ObjectiveOperations is an explicit callable dependency record; runtime integration
comes next. All new device behavior remains unverified on hardware.

## Changes

- `device_normal.py`: Normal schema, offset-aware normalized initialization,
  unweighted gradient/Fisher and weighted NLL. Per-row float64 intermediates must
  produce representable float32 geometry and positive scale/Fisher; no clipping.
- `device_objectives.py`: target/raw widths, K-column broadcast, objective
  dependency bundle, diagonal directions and once-weighted direction fields.
- `_device_kernels.py`: new kernels only; existing scalar kernel bodies unchanged.
- 23 collected real-CUDA checks and ten CPU import/floor checks. Existing input
  ownership and atomic scratch cleanup are reused.

## Verification

- First `test_device_normal_api.py` run failed at import before implementation.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_device_normal_api.py tests/v1/test_device_api.py tests/v1/test_normal_precision_reference.py -n 0 -q`
  — 40 passed.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`
  — 1392 passed, one Linux-only skip; macOS/Python 3.12.12. Local log:
  `/tmp/openboost-090-b-cpu.log` (not a committed benchmark artifact).
- `uv run --no-sync pytest tests/v1/test_device_normal_cuda.py --collect-only -n 0 -q`
  — 23 collected, zero hardware executions. Same cache environment as above.
- Ruff for production/new tests and `uv run --no-sync mkdocs build` pass. Staged
  diff inspected before commit. No GPU simulation substitutes for validation.

## Failed Attempts

- Initial code multiplied unnormalized weight by relative precision. Review
  found representable inputs (`weight=1e30`, offset log scale `-345`) whose
  product exceeds float64 even though normalized initialization is finite.
  Normalize first, preserve the same equation, and retain a hardware check.
- An early generic direction implementation allowed a zero metric when damped.
  CPU semantics require positive diagonals, including ordinary mode. Reject zero
  before output validation; two hardware cases cover that boundary.

## Risks and Follow-ups

CUDA compilation, numerical parity, real transfer counts and resource cleanup
are pending the eventual approved hardware run. Next 090-C generalizes the
existing runtime to mapped terms and K-column transactions; do not add a second
Normal trainer. Known near-tie structural ambiguity remains diagnostic, not a pass.
All hardware/upload allowances remain consumed; no remote action occurred.

## Commits

- `f75f501` — numerical preregistration.
- This entry accompanies the 090-B resident operation construction.
