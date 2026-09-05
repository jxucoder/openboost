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

- Full CPU regression was started; its status and any limits are recorded below.

## Risks and Follow-ups

- Real device result pending at implementation commit. A skipped test is not a pass.
- Wrapper checks are not a profiler trace. Record profiler availability honestly.
- Independent P6 packages still CPU-only; subsequent work must implement/test their
  GPU paths as installed wheels. Eval/early stopping remain CPU-only.
- Device defensive copies and independent traversal can be expensive; measure
  before any speed/cost/value claim. No external adoption claim.

## Commits

- `a9d34f5` — preceding CPU independent wheel evidence.
