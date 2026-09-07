# 2026-09-07: Resident named fields and routed aggregation

## Context

Sprint 086 freezes 078-A's next operation boundary after verified CUDA storage.
The public CPU path once-weights objective fields and preserves independent
information on original routed rows. These semantics must survive device execution.

## Decision or Result

First implement the fixed eight-row and 8192x32 fixtures and a float64 original-row
loop oracle independent of production kernels. Compare the same inputs with public
CPU fields/histograms. Keep empty, unsorted, zero-weight and all-missing subsets.

## Changes

- tests/v1/reference/device_histogram.py records the frozen inputs and loop oracle.
- tests/v1/test_device_histogram_reference.py checks public CPU compatibility and
  hand-calculated totals before any device aggregation implementation.

## Verification

`uv run --no-sync pytest tests/v1/test_device_histogram_reference.py
tests/v1/test_public_numeric_ops.py -n 0 -q`: 25 pass, including seven frozen
fixture cases. Changed-file lint and whitespace checks pass. The public device
import fails with ModuleNotFoundError before implementation. No device aggregation
has run; the remaining T4 allocation is unchanged.

## Failed Attempts

The device field/aggregation API is absent at this baseline; implementation follows
the fixture freeze. Later failed device cases must remain in the raw run record.

## Risks and Follow-ups

Provide public records through DeviceOperations with opaque context-owned storage,
explicit identity/roles and compact validation exports. Histograms reduce actual
rows. Public custom-kernel and accepted-state mutation contracts remain later work;
these primitives do not establish GPU training or D2 candidate feasibility.

## Commits

Fixture/oracle commit precedes implementation and remote execution. No push.

## Device implementation and local validation

DeviceOperations now composes prepared numeric data, named float32 fields,
once-only weighting, resident independent-column append, original-row selections
and padded histograms. Opaque records are bound to one instance. Borrowed input
buffers and owned outputs have different release semantics; empty row views have
zero length. Failed operations discard partial new allocations without modifying
existing inputs. This is not accepted/proposal transaction conformance.

Initial numba-cuda kernels reduce each histogram cell in row order, with float32
accumulation and int64 counts. All arrays use the private CuPy pool; same-stream
CAI adaptation uses as_cuda_array(sync=False), consistent with the
[documented view/stream semantics](https://nvidia.github.io/numba-cuda/user/memory.html).
No global synchronization configuration changes. Compact validation flags are
exported explicitly; kernel dispatch/JIT host time is not GPU elapsed time.

21 focused local checks pass; full CPU regression is 1146 passed, one Linux-only
skip (`/tmp/openboost-078-aggregation-cpu.log`). Lint passes. Thirty-three CUDA cases
collect locally, including twelve storage regressions; collection is not device
validation. CPU tests neither emulate nor skip their way through the new kernels.
The fixture commit is `6ae51dd`. Next freeze/validate the run-2 harness and execute
once from clean source; only actual device results can validate these operations.

## Run-2 freeze and pre-dispatch reflection

The installed-wheel harness pins 33 expected cases, all runtime/build packages,
T4/900-second function/600-second tests/zero retries and 16-MiB private pools.
Its separate JUnit judge rejects missing, duplicated, skipped, errored or failed
cases. Installed production hashes and dependency versions must also match.
Eight focused judge/manifest checks pass; the dirty-source guard fails before
creating output or dispatching. Harness lint and whitespace checks pass.
This is preparation, not a second device invocation or an independent author run.

Reflection after the fixture, implementation and harness slices: the public device
API now expresses the existing named-statistics contract without a private trainer.
D2 can append independent named columns; no candidate feasibility, custom-kernel
registration or accepted-state API exists yet. The kernels deliberately prioritize
simple auditable reductions, so there is no speed expectation. Reference diagnostic
exports are separate from the operation's compact flag exports. No scope or E1
tolerance changed. Execute the frozen run once, preserve its outcome and stop for
the user retrospective; there is no remaining uncounted retry.
