# 2026-09-05: Independent CUDA extension wheel boundary

## Context

P5 verifies strict GPU training through the built-in adapter and an external
builder. The two actual independent packages still declared CPU only. Complete
the GPU part of P6 without modifying the core trainer or importing private APIs.

## Decision or Result

Version 0.2.0 of both example wheels declares CPU/CUDA support, with lazy CuPy
imports and optional CUDA dependency extras. Normal/Fisher formulas are unchanged;
step uses the explicit execution context and rejects host/mixed CUDA inputs.
Initialization stays CPU; loss is an explicit scalar; constrain preserves device.

## Changes

- Normal objective dispatches all vector math through NumPy/CuPy. Reject zero
  precision caused by extreme scale underflow, as well as non-finite statistics.
- BoundedNewton uses the existing public Newton rule and context-array clipping;
  no core edits. Demo accepts --device cpu/cuda and reports actual execution.
- Foundation extensions suite installs all three independently built wheels in
  a source-free container. Hashes cover wheels and package source; compare each
  installed Python file with the wheel. Test math against independent float64
  NLL/Fisher, then 16/4097-row weighted two-round fits with A, A+C, A+B and A+B+C.
- Inspect clipping effects on leaves and next gradients, schedule effects on
  predictions, CPU/CUDA raw/NLL/CRPS and named compact transfers. Execute public
  GPU demo in a subprocess; uninstall both packages, restart Python and require
  nine exact CPU prediction roundtrips with neither plugin importable.
- Runner rejects missing uninstall/independent-inference evidence even if pytest
  passed. CPU wheel conformance remains a separate fresh-venv check.

## Verification

- Initial installed CPU wheel run failed both new CUDA capability declarations.
- Updated CPU installation: 7 tests passed plus weighted public demo; six exact
  CPU model roundtrips after uninstall. CUDA verification follows a clean commit.
- Related CPU regression: 90 passed. Production/example/harness lint and MkDocs
  build passed (existing griffe warnings).

## Failed Attempts

- While adding finite-scale negative tests, identified precision underflow to zero
  for log_sigma=1000. Reject it explicitly instead of accepting zero curvature.
- Lint required explicit binding of y/weights in the metrics helper; fixed the
  test closure before device execution.

## Risks and Follow-ups

- Real T4 package execution pending at implementation commit; no skipped pass.
- Repository-maintained examples are not third-party adoption. No algorithm
  novelty, held-out quality, speed or cost advantage is established here.
- P5 device copies/scalar synchronization and profiler gap remain. Next after
  P6 is P7 measured quality/performance/value, with failures retained.

## Commits

- `baf029d` — preceding strict CUDA trainer evidence.
