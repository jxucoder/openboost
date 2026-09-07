# 2026-09-07: Explicit CUDA storage ownership

## Context

The 078 audit identified immutable host bytes and per-candidate CPU objects as
barriers to resident execution. Begin with actual device storage ownership.

## Decision or Result

Add ExecutionContext with a private CuPy pool and nonblocking owned stream.
Opaque frozen identity handles do not expose arrays. Explicit upload snapshots
host input; device copy owns independent storage; export is blocking and detached.
Reject foreign/forged/released handles, cross-thread operations, invalid limits
and CPU fallback. Release/close synchronize before invalidation. Existing CPU
records, recipes and transaction semantics are unchanged.

CuPy has no NumPy-style writable flag for immutable device views, so a frozen
Python record alone is insufficient. See [CuPy ndarray](https://docs.cupy.dev/en/v13.4.0/reference/generated/cupy.ndarray.html)
and [stream semantics](https://docs.cupy.dev/en/v13.3.0/user_guide/basic.html).
Do not expose authoritative accepted storage through future callback inputs.

## Changes

- execution.py implements storage only, with logical byte counters and sampled
  private-pool peaks; its limit is not whole-device/driver memory enforcement.
- CPU-side validation tests fail initially because the API is absent.
- Real-CUDA tests cover host/export mutation isolation, independent copies,
  stream restoration/order, dtype/NaN bit preservation, handles, budgets and lifetimes.
- cuda_storage_preflight uploads allowlisted source/tests, installs the wheel and
  runs one T4 function. Python 3.12, CuPy 13.6.0, NumPy 2.3.5 and CUDA 12.6.3 image
  are pinned. numba-cuda 0.27.0 remains the lock's planned kernel dependency and is
  not needed for this copy-only slice.

## Verification

1135 CPU tests pass, one Linux-only skip; CUDA-marked tests are excluded, not
counted as passing. Lint/docs checked before commit. The first real T4 run follows
from clean source, capped at 900 seconds with a 600-second test timeout and zero
retries. This is run 1 of the two-run 085 feasibility allowance, even if it fails.
Maximum fixture is 8192x32 float32; private context limit is 16 MiB. Copies must
compare exactly; this has no reduction tolerance or training metric claim.

## Failed Attempts

Initial API import fails before implementation. Remote outcomes are recorded
separately. A missing GPU is not success or a CPU emulation result.

## Risks and Follow-ups

No device fields/histograms/boosting or accepted-state integration yet. Transfer
counts cover this API only; pool peaks exclude driver and outside allocations.
Next add named fields and routed reductions under these ownership contracts.
Do not consume another device run without preregistering its concrete fixtures;
two-round scalar training, D2, P7 and formal E4 remain separate requirements.

## Commits

Implementation commit precedes device execution. No push.
