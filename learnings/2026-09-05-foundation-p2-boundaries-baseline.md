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

## Failed Attempts

- Automatic review rejected the P2.2 Modal command because the new test file
  was outside the previously approved manifest, despite the wheel being
  byte-identical to the previously approved wheel. No remote job started.
  Finish local work and request explicit approval for the complete new bundle;
  do not bypass the rejection by rerouting execution.
- Sandboxed public-data download failed DNS. A reviewed network-only download
  succeeded and its archive hash verified. No project content was uploaded.
- Lint rejected a lambda assignment in the worker; replaced it with a function.

## Risks and Follow-ups

- P2.2 and P2.3 GPU runs are still pending the new bundle upload approval. The
  locally prepared harness is not proof that CUDA boundaries or quality pass.
- The trainer download spy/counter covers only named trainer boundaries;
  compact tree conversion and backend internals can still copy. No complete
  PCIe accounting, GPU peak memory measurement or zero-transfer claim.
- Per-seed NLL difference <= .01*max(1,abs(CPU NLL)), CRPS regression <= 1%, and
  coverage90 difference <= .01 follow the design. They cannot waive strict
  micro-oracle failures, and three seeds are not a significance claim.

## Commits

- `576702d` — P2.1 before/after T4 evidence.
- `4423117` — eight-case execution boundary harness, not yet GPU-validated.
