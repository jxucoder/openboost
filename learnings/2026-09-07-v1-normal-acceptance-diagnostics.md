# 2026-09-07: Normal acceptance needs differences at stored inputs

## Context

The user approved continuation after run 6's two ordered Normal acceptance
failures. [Sprint 091](../v1-sprints/091-normal-acceptance-diagnostics.md) starts
with local diagnostics. Six hardware allowances remain consumed; this is not
another upload/run approval.

## Decision or Result

An independent Decimal oracle evaluates the Normal NLL difference directly from
original rows, with exact binary-to-decimal input conversion. At 60 and 100 digits,
both adjacent float32 mean states around the saved passing D2 base worsen loss.
The old float64 full-loss subtraction reports both as improvements by one ULP.
A separate analytic case also shows that equal rounded full losses can conceal a
real `-2^-61` improvement. Neither result justifies a blanket epsilon.

This falsifies universal reliability of full-loss subtraction near stationarity,
not the device kernel itself. The missing failed-state trace must be measured
before diagnosing either original failure or selecting production semantics.
The precision comparison is numerical evidence, not an interval proof.

## Changes

- Independent [oracle](../tests/v1/reference/normal_acceptance.py) and fourteen
  CPU checks for analytic differences, weights, offsets, invalid inputs and the
  saved-base counterexample. Existing reference files are unchanged.
- [Reproducible study](../benchmarks/v1/analyze_normal_neighbors.py) retains input
  and source hashes, loss/raw bits, both precision estimates and CPU environment.
  It verifies the saved model against the immutable run-6 manifest. It executes
  neither CUDA nor an emulator. A generated artifact follows a clean commit.

## Verification

`OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync
pytest tests/v1/test_normal_acceptance_reference.py tests/v1/test_cuda_normal_manifest.py
-n 0 -q` — **26 passed**. The twelve archive checks preserve run 6 and all nineteen
saved-model CPU replays. Ruff passes for the three new Python files.

## Failed Attempts

The first tiny-improvement test passed a list offset to the older array-indexing
reference and raised TypeError. Corrected the new test's input representation;
no oracle arithmetic, existing test or production source changed.

## Risks and Follow-ups

Next add real-device observation wrappers and offline trace analysis. Preserve
all 383 original tests and the strict all-pass judge. A complete diagnostic trace
will not repair conformance. Freeze the package before requesting another upload
and hardware allowance. Normal/P7/E4 and all remaining application/author gates
stay open; no push occurred.

## Commits

- `abd5b1a` — prior run-6 retrospective and proposed investigation.
- `ba46b3a` — independent loss-difference oracle and local sprint plan.

## Observation construction and local verification

The [neighbor artifact](../benchmarks/v1/evidence/normal-acceptance-local-091/README.md)
was generated at clean `ba46b3a`; its mathematical results reproduce exactly.
The separate device diagnostic calls the original failing test with temporary
wrappers around initialization, growth, geometry/direction, field/leaf operations
and proposal/resolve. It captures values and bits before the original assertions,
then restores every patched method. Only the original coefficient-list assertion
is retained as a known conformance failure inside a successful measurement;
unexpected assertions/exceptions propagate. Partial JSON survives either path.

Read-only diagnostic access to private prepared target/offset handles and the base
is deliberate: these records lack a public export operation. It establishes actual
stored inputs instead of assuming a host cast matched preparation. No production
API is expanded for diagnostic convenience. All other raw snapshots are public
owned copies. Guards check buffer/record sets, run serial, state/proposal ownership,
live bytes and uploads around observation; actual device behavior remains unrun.

Source inspection caught two harness errors before hardware: Normal initialization
calls geometry before the run is initialized, and field roles describe weighting
while field names identify gradient/curvature. The wrapper now captures constructor
geometry separately; offline analysis indexes named fields. A CPU analytic record
with reordered columns verifies that distinction. Synthetic analysis-test records
are explicitly labeled and never used as device evidence.

The offline analyzer compares captured gradients/Fisher and reductions to original
rows at exactly the stored inputs, and loss differences at both precisions. It
checks bit integrity, trial halving/parents, strict acceptance, state continuity,
rejection identity, accepted snapshots and independent best-prefix selection.
Observed values never drive the original algorithm's decisions.

Focused verification: the same CPU command now includes
`tests/v1/test_normal_acceptance_trace.py` and passes **45 tests** (15 oracle/study,
18 analysis, twelve archive checks). Both new GPU cases collect with
`uv run --no-sync pytest tests/v1/test_device_normal_acceptance_cuda.py --collect-only
-o addopts= -q`; neither executes locally. Production/changed-file Ruff and MkDocs
pass; MkDocs retains the pre-existing execution-page evidence-link warning.
No production source, old test, tolerance, failed artifact or all-pass judge changes.
