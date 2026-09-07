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
- Remote paired fit and fresh replay passed; details follow below.

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

- `73755b1` — Prepare exact paired A6 real-fit cost check.
- Paired evidence is recorded in the following slice.

### Paired real-fit result and reflection

At clean `73755b1`, one same-container shared A6 pair against original public CPU
source `17ab9de` passes. [Raw evidence](../benchmarks/v1/evidence/a6-paired-070/README.md)
retains 32 current source hashes, 20 baseline source hashes and 40 raw artifact
hashes, all verified. Loaded package paths identify the source baseline and
installed current package. Model/prediction/training/replay bytes are identical
across variants and with the original shared real probe. Both stop after 59 rounds.

Baseline fit: 992.047 seconds; current: 757.480 seconds, an observed 23.6448%
reduction in this single fixed-order pair. Fresh replay: 0.319/0.269 seconds;
peak guest RSS: 107200512/106422272 bytes. Both satisfy the existing 1800-second
fit limit. Host CPU/RAM details are unavailable; requested capacity and guest RSS
remain distinct. No repeated/general speed or total cloud cost claim is made.

Reflection: the measured optimizations now have exact real-fit evidence. End this
optimization detour. Return to complete evaluator-owned search/comparator coverage
and the 069 accounting/isolation packet. Keep the other 158 OpenBoost jobs pending;
deep/1000-round cases and full selection still need their frozen resource checks.
This pair does not pass authoring, full quality/search, adoption or CUDA gates.

Verification: 23 focused contracts and 1095 CPU tests passed with one Linux-only
skip at the harness commit. The actual remote pair and fresh replay now pass;
all source/artifact/input pins verify. No failed attempt or retry in this pair.
