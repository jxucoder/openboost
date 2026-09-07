# Sprint 078: Resident scalar CUDA composition

Status: planned, device work not started. Mapping: B12 / F3.1 / R1 / C1–C5 / E1.
Depends on: 068 state/diagnostic ownership and formal F2 pass from 077, unless
the proposed bounded overlap is explicitly adopted in the main plan first.
Shared evidence/closure rules: [roadmap](roadmap-after-063.md).

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

Not run. Current OpenBoost has no CUDA implementation.
