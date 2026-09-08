# 2026-09-06: Bind current selection to explicit worker permissions

## Context

The actual current selection smoke still inherited evaluator identity after the
standalone Linux permission probe passed. Protocol files and test features shared
its worker-readable output root. Packet schema rejection did not prevent file reads.

## Decision or Result

Extend the existing smoke with explicit protected mode, not a separate executor.
Evaluator records live under root-owned mode 0700; candidate inputs are read-only.
Workers use the existing UID/resource wrapper, and completed trial directories are
reclaimed before subsequent trials. Unsupported hosts fail before writing files.

## Changes

- Preserve validation-only audit, receipt sealing, selected release and fresh replay.
- Record explicit address, identity and timeout policy in environment provenance.
- Keep the worker's supported one-thread contract and public default behavior.

## Verification

Two unsupported-host tests fail before implementation and pass afterward. Local
16-trial smoke passes with selected ID `openboost:15`; existing independent audit
and fresh selected prediction checks pass. Full suite: 1021 passed, 1 skipped;
lint passes. The skipped Linux/root test includes actual protected-path fault
operations, real recipe fits and completed-output ownership checks.

## Failed Attempts

The local Mac cannot certify Linux identity changes. The earlier standalone probe
is not evidence that this selection integration works. No such claim is made.

## Risks and Follow-ups

Run the real integration test on Linux with a traversable installed runtime and
retain raw evidence. This is a four-round synthetic grid, not full model resource
qualification or application quality. Network and escaped sessions are not bounded
by this mode; independent hostile author attempts still need separate containers.

## Commits

- Protected selection integration; parent `4a4cc2d`.
