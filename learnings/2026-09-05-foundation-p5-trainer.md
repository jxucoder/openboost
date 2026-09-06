# 2026-09-05: Strict CUDA extension trainer integration

## Context

P4.4 verifies direct GPU builder composition and P6 CPU verifies independent
wheel installation. Actual experimental Booster.fit still ran only on CPU.
Reuse the unified trainer rather than duplicate its round loop.

## Decision or Result

Add a strict device extension session to the shared trainer. CPU initialization
and binning are explicit; CuPy owns raw/targets/weights and device updates.
Default CUDA selects LevelWiseBuilder, while explicit builders keep priority.
CPU defaults remain unchanged. Normal/Poisson adapters declare device support
only when the existing objective exact-type capability permits it.

## Changes

- Device objective bridge validates dtype/device/finiteness/ownership and input
  mutation. Device builder session validates compact standard tree state and
  independently traverses it on GPU to verify optional cached predictions.
- CuPy cannot offer NumPy read-only views, so borrowed plugin inputs are isolated
  by device copies and checked after the call. This includes binned inputs per
  tree and is a known memory/bandwidth cost, not an optimized performance claim.
- Preflight rejects missing/categorical, sampling, L1, eval/callbacks/early stop,
  unsupported device/leaf capabilities, budget and non-default stream. Warn
  fallback selects the full CPU path; runtime errors use existing rollback.
- Report actual stages, CPU binning/init and scoped transfer/copy behavior.
  Model persistence retains host binner/tree state; prediction remains CPU.
- GPU harness exercises default/explicit dispatch, weighted two-channel two-round
  schedule, CPU/CUDA raw/NLL/CRPS, CPU load and failure rollback with blocked
  legacy dispatch and named host-download wrappers.

## Verification

- New CPU preflight tests first failed against the CPU-only facade.
- Focused preflight, evidence runner, objective, dispatch, persistence and builder:
  89 passed. Real T4 validation follows after a clean implementation commit.

## Failed Attempts

- Full CPU regression completed during implementation: 904 passed, 34 skipped,
  20 deselected in 325.15 s. No failures. The final targeted preflight/extension
  run after the last preflight edits passed 89 tests; the expanded evidence runner
  passed 20. Lint and MkDocs build passed (existing docstring warnings).

## Risks and Follow-ups

- Real-device core integration and declared adapter modes passed below.
- Wrapper checks are not a profiler trace. Record profiler availability honestly.
- Independent P6 packages still CPU-only; subsequent work must implement/test their
  GPU paths as installed wheels. Eval/early stopping remain CPU-only.
- Device defensive copies and independent traversal can be expensive; measure
  before any speed/cost/value claim. No external adoption claim.

## Commits

- `a9d34f5` — preceding CPU independent wheel evidence.

## Initial T4 result and coverage expansion

- `8c34b26` actual trainer suite: 7 passed / 0 skipped; raw max error 1.19e-7,
  matched NLL/CRPS and CPU save/load, runtime/input/cache failure rollback.
  [Initial artifact](../benchmarks/results/foundation/20260905T175651Z-b1f9743a/README.md).
- Profiler gap confirmed: nsys unavailable. Named transfer checks only.
- Follow-up test coverage verifies all advertised adapter modes (Normal/Poisson,
  ordinary/natural), a duck-typed external builder and invalid dtype/Hessian/alias
  outputs. This broadens checks without changing the implementation or tolerance.

## Final declared-surface evidence

- `4299858`: [final T4 artifact](../benchmarks/results/foundation/20260905T180000Z-7d73ba83/README.md),
  7 passed / 0 skipped; Normal/Poisson × ordinary/natural modes and external
  duck-typed builder all pass. Maximum raw CPU/CUDA error 1.19e-7; NLL/CRPS agree.
- Invalid dtype, negative Hessian and output alias checks pass in addition to
  runtime, mutation and cached-update failures. Source/lock hashes and JUnit
  independently verified; offline validator and privacy scan passed.
- Independent CPU wheel conformance also reran: 5 passed, weighted demo passed,
  six exact predictions after plugin uninstall. That local result is marked
  dirty because the initial GPU evidence directory appeared during the run;
  its wheel hash exactly matches the clean T4 implementation wheel. Do not
  present it as a separate clean-source evidence artifact.
- P5 core execution gate passes with the documented profiler gap (nsys absent).
  Next: P6 actual installed GPU extensions, then P7 profiling/quality/cost.
