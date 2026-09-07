# 2026-09-07: Paired A6 real-fit checkpoint

## Context

Two scoring changes remove measured redundant CPU work, but separate 60-second
profiles cannot establish complete-fit cost. Sprint 070 requires a paired check
before further optimization or search expansion.

## Decision or Result

Compare original public CPU source at `17ab9de` against current installed source
in one container, baseline first. Only ops.py differs in production. Require exact
model, prediction, training/stopping and fresh replay bytes. Record loaded package
paths to distinguish source baseline from installed current code. Keep dependencies,
worker, input and configuration identical. Fixed order and one observation limit
interpretation; neither comparative quality nor general speed is established.

## Changes

- `benchmarks/v1/a6_resource_preflight.py`: mutually exclusive paired mode, frozen
  public baseline source upload, sequential fit/replay, strict artifact equality,
  fresh replay timing and retained timeout failures.
- `tests/v1/test_a6_paired_contract.py`: reject changed/missing/empty evidence,
  failed execution and replay; verify first-failure stop and artifact retention.

## Verification

- Focused paired/profile contracts: 23 passed.
- Full CPU regression: 1095 passed, one Linux-only skip. Lint and docs pass.
- No paired remote fit has run yet.

## Failed Attempts

None in this slice. The previous profiling failures remain in their own evidence.

## Risks and Follow-ups

One real shared fold-zero configuration only, 300 rounds with patience 50. Each
fit has 1800 seconds and 8 GiB address space; replay has 60 seconds. One Modal
function has 3800 seconds, two requested CPUs, 8192 MiB and no retries. Baseline
source is explicitly imported; current is installed. Validate actual loaded paths
and all source/artifact hashes after execution. The remaining A6 matrix, comparator
coverage, authoring and CUDA gates stay open.

## Commits

Implementation commit follows verification; paired evidence is a separate slice.
