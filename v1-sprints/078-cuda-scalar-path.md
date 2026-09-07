# Sprint 078: Resident scalar CUDA composition

Status: 078-A/B primitives verified on T4 at `9ce790e` with 88 passing tests;
resident training and transactions remain open. All three approved invocations
are consumed; planned retrospective reached. Mapping: B12 / F3.1 / R1 / C1–C5 / E1.
Entry: bounded feasibility is approved by [085](085-foundation-focus-amendment.md)
alongside 069, subject to relevant 065/068 ownership checks. Formal device gates
still retain their full recipe, quality and authoring requirements.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).
Next construction slices, frozen aggregation fixtures and run-2 stop boundary:
[086 execution plan](086-next-execution-plan.md).

## Outcome and first failing check

Run a public squared-error recipe on one real CUDA device using resident bulk
operations. First use a two-round weighted numeric/missing fixture that currently
fails because device execution is unsupported; compare each stage to independent
CPU references, not array shapes or final score alone.

## Work

- Pin CuPy/numba-cuda, hardware/driver, precision and tie policies, shape/round
  limits, hard time/memory/compute budget and compilation policy before a bounded
  Modal experiment. Keep other feasibility hardware separate from frozen T4 E4.
- Implement explicit device/workspace/stream ownership and bulk fields, candidates,
  routing and leaf operations under existing semantics. Separate small host
  metadata from device buffers; do not traverse host candidate objects with transfers.
- CPU preparation plus one upload is allowed with costs recorded. Keep repeated
  gradients, aggregation, routing and raw updates on device with no silent bulk
  CPU fallback. Unsupported capabilities fail explicitly.
- Preserve atomic acceptance, best/stop, scoped RNG and CPU-readable export.
  Expose transfers, synchronization, rejected work, JIT and workspace use.
- Prepare the historical P7 protocol, fixed data/configuration and comparison
  environment for 079. The [P7 workload](../benchmarks/results/foundation/20260905T193308Z-5ebd75ab/README.md)
  uses Normal distributions, so its actual reproduction needs 079's multiparameter
  path. A scalar fit cannot substitute for that separate original 1.2-threshold check.

## Acceptance and reflection

At least two actual rounds agree on gradients/curvature, named statistics, candidate
optimality/ties, routes, leaves, raw state, predictions and task metrics under E1.
Counts/routes are exact; float32 intermediates use the frozen tolerances and tie
policy. Fresh CPU inference reads the artifact without CUDA or training plugins.

Retain full end-to-end costs and memory provenance, including failure statuses;
an unavailable GPU is not_run. Advance to 079 only after scalar correctness/residency
passes. If the implementation needs a private bypass or full host ensemble replay,
revise that boundary before adding kernels. No general GPU-speed claim yet.

## Results

Storage-only implementation and real-device results are recorded below. The
two-round scalar training acceptance has not run or passed.

### Ownership audit

The [actual CPU seam audit](078-device-boundary-audit.md) identifies forced NumPy
storage, host candidate callbacks and CPU accepted-state caches. First implement
explicit execution-owned buffers/upload/export, then named fields and histograms;
keep accepted storage separate from extension workspace. No CUDA implementation
or hardware test was claimed by that initial audit. The later storage result
below supersedes its implementation status; the operation/state gaps remain.

### Execution-owned storage implementation

Plan: first fail import/validation for the absent execution API; implement opaque
context-owned buffers with explicit upload/copy/export/release; verify validation
locally and ownership/lifetimes on one real T4. Use CuPy 13.6.0 from uv.lock, NumPy
2.3.5 and Python 3.12. The first 085 allocation is one function capped at 900 seconds,
no retries, at most 8192x32 float32 test data and 16 MiB per-context pool limit.
This consumes one of the two allowed device runs even on failure. Exact array
copy equality is the oracle; no floating reductions or tolerance is needed yet.
Do not call storage checks two-round boosting or transaction conformance.

### First real device result

[Storage evidence](../benchmarks/v1/evidence/cuda-storage-078/README.md) at `77aa105`
passes twelve installed T4 tests with exact copies, lifetimes, context/thread
checks and visible transfer/pool accounting. All 26 source hashes and the log
verify. The fixture private-pool peak is 2 MiB, not total GPU memory. Actual CUDA
runtime/driver API versions are 12090/13000 despite the 12.6.3 image tag.

085 budget: run 1 consumed, no retries; one further 900-second run remains. Freeze
its field/reduction fixtures before dispatch. Next build named device fields and
routed histograms with D2 information semantics. Two-round scalar training,
accepted/proposal integration and CPU-readable trained artifacts remain open.

### 078-A fixture freeze

The 086 eight-row and 8192x32 inputs now have a float64 original-row loop oracle
in tests/v1/reference/device_histogram.py. Seven fixture cases and eighteen public
CPU operation tests pass before device implementation; hand totals include
[10,9,4,4] and zero-weight-only [0,0,2,0]. The new public device import fails as
absent. Next implement DeviceOperations fields/rows/histograms. No additional
GPU run is consumed by these local reference checks.

### 078-A implementation awaiting real-device validation

DeviceOperations provides public prepared data, named fields, once-only weighting,
independent-column append, resident/host row views and routed histograms. Initial
numba-cuda reductions use the owned CuPy stream/pool and export compact validation
flags. Partial allocation/validation failures preserve existing input buffers.
No custom-kernel registration, candidates, gradients, trees or GPU training yet.

Local checks: 21 focused pass; full CPU suite 1146 passed, one Linux-only skip.
Thirty-three real-CUDA cases collect but have not executed. See the
[learning record](../learnings/2026-09-07-v1-device-aggregation.md) and public
[execution docs](../docs/v1/execution.md). Run 2 remains available pending the
committed installed-wheel harness and expected-case freeze.

### Run-2 dispatch freeze

[078-aggregation-run2.json](078-aggregation-run2.json) fixes the 33 expected tests,
dependency versions, bounds and unchanged E1 tolerances. The harness uploads only
listed package/test/oracle/config files, installs the wheel, checks exact installed
sources/versions and judges every expected JUnit case. Eight local judge/manifest
checks pass; the dirty-tree guard rejects before output or dispatch. Tests include
the existing twelve storage checks and 21 aggregation/identity/failure cases.

Reflection after three construction/preparation commits: named field semantics
and lifetimes are explicit, with no bulk CPU histogram work; actual CUDA behavior
is still unverified for this implementation. D2 candidate selection and training
remain open. Run this clean revision once on T4 (900 seconds, zero retries), then
retain pass/fail/error/timeout artifacts and stop for a retrospective. This is the
last run in 085's allowance; neither a failure nor an incomplete test matrix
authorizes a retry. The [learning record](../learnings/2026-09-07-v1-device-aggregation.md)
retains local verification and public-boundary limits.

### Run-2 result and retrospective checkpoint

[Committed evidence](../benchmarks/v1/evidence/cuda-aggregation-078/README.md): all
33 expected cases pass at clean `ad2f4e6`. All 35 snapshot hashes, 23 installed
production modules, 17 pinned versions and three result-artifact hashes verify.
The real T4 reports driver 580.95.05, runtime API 12090 and driver API 13000.
The full 8192x32 aggregation context records a 1665024-byte private-pool peak,
not whole-device memory. Each four-field histogram exports only 32 validation
bytes during the operation; reference exports are separate. There are 31 retained
low-occupancy warnings, not evidence of end-to-end speed. No retry occurred.

078-A passes its frozen acceptance. Both 085 device runs are now consumed. Stop
here for the planned user retrospective; no third run is authorized. Next local
construction is 078-B's candidate batches, feasibility, routes and leaves, starting
with exhaustive scalar/D2 fixtures and an explicit public component contract.

What changed: the foundation now performs useful named reductions on actual CUDA
with original-row/weight semantics. What did not change: it cannot train on GPU,
select a D2-constrained split, or accept arbitrary external kernels. Opaque handles
protect storage but do not by themselves solve public device programmability.
That interface and later accepted-state ownership must be consumer-tested before
claiming an agent can build a full custom GPU boosting algorithm.

No independent author attempt or accounting result was added. Wider CPU searches
stay paused. Formal application quality, author benefit, P7/E4 and adoption remain
open. The next remote verification needs a concrete reviewed workload and a new
bounded allowance; this successful allocation is not a reusable compute credit.

### Run-3 public split result and retrospective

The separately approved [087](087-cuda-split-operations.md) completes 078-B:
55 new split cases and all 33 earlier regressions pass at clean `9ce790e` on T4.
The [raw evidence](../benchmarks/v1/evidence/cuda-splits-078/README.md) verifies
39 snapshot files, 23 installed production modules, 17 pinned dependencies and
the exact 88-cell JUnit matrix. The approved extra invocation is consumed; no
retry occurred. CPU recipes and formal phase status are unchanged.

Independent candidate sums/counts/gains/masks, exact tie selection, original-row
routes and scalar leaves agree. D2's best split changes through public independent
cohort minima, including renamed/reordered fields and changed minima. No host
candidate callback or bulk array round trip implements this constraint.

This verifies the primitive boundary, not resident boosting. Next freeze the
078-C accepted/proposal ownership contract and independent two-round fixture;
keep device raw updates, rejection/retry isolation, best/stopping separation and
CPU-readable saved inference in scope. Current opaque handles and operation
allocation cleanup do not pass the 065/068 transaction/retention requirements.
Keep author accounting, full application quality, P7/E4 and adoption gaps visible.
Stop for the planned retrospective before constructing the next slice.
