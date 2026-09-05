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

- Real T4 package conformance passed below; no skipped GPU tests.
- Repository-maintained examples are not third-party adoption. No algorithm
  novelty, held-out quality, speed or cost advantage is established here.
- P5 device copies/scalar synchronization and profiler gap remain. Next after
  P6 is P7 measured quality/performance/value, with failures retained.

## Commits

- `baf029d` — preceding strict CUDA trainer evidence.

## First real-device attempt

- `5d10b77`: [failed artifact](../benchmarks/results/foundation/20260905T180943Z-e145df4a/README.md),
  1 failed / 2 passed. GPU finite-difference/Fisher and eight training cells
  reached the public demo call successfully, but the subprocess failed CUDA
  availability discovery with a multiprocessing traceback. Demo fitting at
  module top level lacked a main guard and could re-enter in spawned workers.
- Added a main entry guard, a no-side-effect __mp_main__ import check in the CPU
  installer, and full subprocess stderr on failure. Retain original failure;
  rerun rather than suppress the CUDA failure or weaken parity tolerances.

- Entry-guard fix verified locally: 7 fresh-wheel tests and six plugin-free CPU
  roundtrips, plus no-side-effect worker import; evidence runner 21 passed.
  Preserve partial GPU metrics before launching the demo on future failed runs.

## Frozen successful evidence

- `43fcda3`: [passing T4 artifact](../benchmarks/results/foundation/20260905T181351Z-1aee9568/README.md),
  3 passed / 0 skipped. Independent installed GPU math plus eight combination
  cells and standalone GPU demo; nine exact CPU predictions after both plugins
  were uninstalled and a new interpreter started.
- Maximum raw error 3.58e-7, NLL difference 2.31e-8, CRPS difference 2.86e-8.
  These are numerical fixture agreement, not held-out quality or speed evidence.
- Named compact copies: 160 / 4,480 bytes over eight GPU fits; separate demo and
  reference transfers excluded. Device copies/scalar synchronization remain.
- Main-guard fix resolved the standalone CUDA discovery failure. Preserve its
  original JUnit whitespace verbatim; the commit's whitespace check excluded
  only that immutable traceback file, not implementation files.
- All uploaded file, package source, wheel and lock hashes verified; JUnit copies
  matched; strengthened offline result gate and private-path scan passed.
- P6/G3 technical package gate passes. Next: P7/G4 matched-quality cost and
  developer materials; G5 remains open until an external author's actual use.
