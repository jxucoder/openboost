# Sprint 078: Resident scalar CUDA composition

Status: experimental storage verified on T4; resident fields, operations and
training remain open. Mapping: B12 / F3.1 / R1 / C1–C5 / E1.
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
