# 2026-09-05: P2 execution boundaries and baseline

## Context

P2.1 passed on T4 at 8609f70. Continue with execution boundaries and the planned
real-data baseline, preserving wheel-only provenance and historical artifacts.

## Decision or Result

A separate boundaries suite preserves historical correctness-test semantics.
Eight new GPU cases join the five existing ones: custom same-name distribution,
exposure, generic tree fallback, runtime error rollback, two sampling preflights
and two callback/eval/persistence cases. No production code changed here.

Baseline configuration and gates are frozen before any GPU baseline result.
California Housing uses sklearn 1.8.0's archive URL/SHA-256 and transformations.
Its archive, float32 arrays and seed 0/1/2 splits have committed hashes in
`benchmarks/foundation/housing.json`. The public archive is stored in ignored
`build/foundation_data/`, never silently replaced with synthetic data.

The baseline model uses 30 rounds, depth 3, learning rate .05 and 64 bins.
Each 60/20/20 split has 12,384 training and 4,128 validation/test rows. No learned
scaling; only training data fits bins. CPU/CUDA × three seeds × no-eval/eval
produces 12 fresh subprocesses, two fits each. Every process gets a fresh
NUMBA_CACHE_DIR; first/repeat fit times include binning, objective math, copies,
compilation and path instrumentation, but exclude imports/loading/startup.
CUDA reductions use rtol=2e-5/atol=2e-6 repetition checks, not bitwise identity.

A single baseline Modal run first executes all 13 correctness/boundary cases
with maxfail=1, and only then the matrix. Per-cell runtime limit is 150s,
pytest limit 1740s and function limit 1800s, one T4, CPU=2, memory=8192 MiB,
retry=0. It preserves partial cells and failure output. This is a bounded
baseline, not a scaling experiment or a speed claim.

## Changes

- `tests/foundation/test_boundaries.py`: actual CUDA execution and visible
  fallback; CPU comparisons; GPU-save/CPU-load and reverse round trips.
- `dataset.py` / `housing.json`: public archive loader and frozen data/splits.
- `baseline_worker.py`: end-to-end fit/predict, independent Normal metrics,
  cold/repeated fit policy and execution-path instrumentation.
- `test_baseline.py`: ordered matrix and predeclared NLL/CRPS/coverage gates.
- Preparer/Modal/runner: allowlisted baseline bundle, complete matrix gate,
  CPU model provenance when exposed, and separate timing scopes.
- `tests/test_foundation_baseline.py`: missing/duplicate/fallback/device/metric/
  timing/quality rejection, split isolation and independent metric oracle.
- `benchmarks/foundation/README.md`: protocol, reproduction and limits.

## Verification

Local commands use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.

- `OPENBOOST_BACKEND=cpu uv run --no-sync pytest tests/test_foundation_baseline.py
  tests/test_foundation_runner.py -n 0 -q`: **27 passed**.
- `uv run --no-sync ruff check benchmarks/foundation tests/foundation
  tests/test_foundation_baseline.py tests/test_foundation_runner.py`: passed.
- Isolated pytest collection verifies 14 cases in order: smoke, weighted
  correctness, eight boundaries, then baseline. Collection is not a GPU pass.
- Downloaded archive SHA-256 matches the pinned sklearn value. Its loader output
  exactly matches sklearn 1.8.0 after float32 conversion (all features/targets).
  The reference fetcher received a local verified copy, avoiding another download.
- Local real-data CPU seed-0 resident and eval development cells completed;
  first/repeat and eval/no-eval predictions were bit-identical, and evaluation
  logged all 30 rounds. No development timing is promoted to baseline evidence:
  these checks ran before the harness commit and on a different local CPU.

## Real-device verification

- Tested clean source `99621ae9e640be9c1144a015a39de43d3bf50ae9`, same production
  wheel as P2.1; [raw artifact](../benchmarks/results/foundation/20260905T084129Z-2574e387/README.md).
- **14 passed, 0 skipped**, pytest 104.58s, remote function 108.88s.
- All eight execution boundaries passed, including visible custom/exposure/
  generic fallback, failed-kernel rollback, sampling preflight, callback/eval
  and both CPU↔GPU persistence directions for Normal/Poisson.
- All 12 CPU/CUDA × seed × eval-mode cells completed two fits. Maximum held-out
  NLL difference 1.34e-8, CRPS difference 1.12e-8, coverage difference zero.
  No fallback warnings; each GPU fit uses 30 objective calls and 60 native trees.
- Offline validator accepted the result. Raw artifacts contain no local user
  paths or private Modal app URLs. Copied source hashes, frozen dataset/
  split metadata and exact JUnit content independently match the source commit.
- Median repeated fit: CPU 2.401s / GPU .145s without eval; CPU 3.027s / GPU
  .230s with eval. These measurements are scoped in the artifact and do not
  establish a general library-level speed claim.

## Failed Attempts

- Automatic review rejected the P2.2 Modal command because the new test file
  was outside the previously approved manifest, despite the wheel being
  byte-identical to the previously approved wheel. No remote job started.
  The user subsequently explicitly approved the new files. The original Modal
  upload/run command then succeeded; no rerouting was used.
- Sandboxed public-data download failed DNS. A reviewed network-only download
  succeeded and its archive hash verified. No project content was uploaded.
- Lint rejected a lambda assignment in the worker; replaced it with a function.

## Risks and Follow-ups

- P2.2 and P2.3 now pass on the tested T4 source. Next gate is P3, the minimal
  CPU extension contract; do not infer new backend/extension capabilities from
  this existing-trainer baseline.
- Nominal 90% interval coverage is 96.39–96.78% in this untuned baseline.
  CPU/GPU parity is excellent, but calibration and tuned product quality remain
  separate work. First-fit timing does not clear CUDA driver caches; /proc
  exposes CPU model as `unknown`. Neither missing fact is invented.
- The trainer download spy/counter covers only named trainer boundaries;
  compact tree conversion and backend internals can still copy. No complete
  PCIe accounting, GPU peak memory measurement or zero-transfer claim.
- Per-seed NLL difference <= .01*max(1,abs(CPU NLL)), CRPS regression <= 1%, and
  coverage90 difference <= .01 follow the design. They cannot waive strict
  micro-oracle failures, and three seeds are not a significance claim.

## Commits

- `576702d` — P2.1 before/after T4 evidence.
- `4423117` — eight-case execution boundary harness.
- `99621ae` — frozen baseline harness, now verified on real T4.
