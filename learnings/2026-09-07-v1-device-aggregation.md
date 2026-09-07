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
