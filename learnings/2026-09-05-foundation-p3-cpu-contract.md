# 2026-09-05: P3 CPU extension contract

## Context

P2 baseline passed at 99621ae; P3 introduces a small CPU extension facade while
retaining the existing trainer loop. The user requested continuation.

## Decision or Result

Implement in three verified slices: objective/arrays, builder/schedule, then
persistence/early-stop consistency. CPU-only execution is explicit; requested
CUDA either fails preflight or warns and selects the entire CPU path.

## Changes

- `experimental.Booster`, immutable `ExecutionContext`, shared `TrainerConfig`
  and a distribution adapter. The facade invokes the existing trainer.
- Validate complete channel keys, contiguous float32 statistics, nonnegative
  curvature, finite data/weights, positive total weight and explicit devices.
  Inputs are read-only views; aliased outputs fail without per-round host copies.
- Add min_gain to the single trainer config and propagate it to tree builders;
  a supplied per-fit generator can be shared with extension contexts.

## Verification

- Initial independent two-channel objective test failed import before creation.
- The first implementation exposed a shared weight-validator boundary: it
  allows all-zero weights and Inf. The experimental facade now rejects them
  before binning without changing legacy validation semantics.
- Slice 1: 53 passed (23 experimental, plus existing foundation/formula/survival
  tests); production/new-test lint passed. No experimental GPU execution is
  claimed by P3.

## Builder/update slice

- Explicit builder dispatch precedes native eligibility; every channel uses a
  single round-start objective evaluation. Schedule values are full coefficients
  stored alongside trees and reused for prediction/eval.
- Package-owned standard scalar trees are validated before prediction. Compact
  arrays are detached, permitting builder scratch reuse without sample-sized
  copies. Optional cached predictions must match the tree.
- CPU zero-curvature root handling is explicit. The legacy split kernel does
  not define 0/0 candidate scores, so the CPU adapter rejects the unverified
  reg_lambda=0/min_child_weight=0 pair rather than silently changing it. This
  is a declared P3 boundary; future primitive work can broaden it.
- Slice 2: 93 focused/legacy tests passed; production and new-test lint passed.
- Tests hand-check two channels/two rounds, explicit dispatch even when native
  eligibility is true, unhalved split gain, cache consistency and zero curvature.

## Failed Attempts

- All-zero weights initially passed; fixed at experimental preflight.

## Risks and Follow-ups

- A native extension adapter remains future CUDA work; no unverified
  adapter is exported by this CPU contract.
- Read-only views prevent accidental mutation; this is not a sandbox against
  deliberately hostile Python plugins accessing underlying memory.

## Commits

- `4c16204` — completed P2 evidence.

## Persistence slice

- Reuse the existing version-2 tree/binner serializer with an experimental
  version-1 marker and an explicit inference-state whitelist. Training plugins
  and config are excluded; loaded models reject fitting. Missing coefficients
  use the saved constant learning rate, while invalid counts/values fail.
- Early stopping snapshots and restores coefficients alongside trees. Last-round
  and train-end learning-rate mutations now fail for the experimental path.
- Six initial persistence tests failed because Booster had no save/load API;
  checkpoint fell back to pickling the custom objective. All 13 persistence and
  callback-boundary tests now pass, including categorical/missing round trips,
  nonconstant coefficients, unpicklable plugins and invalid versions.
- Shared legacy version-1 categorical loading currently warns; the new facade
  rejects that state without changing legacy policy.
- Regression: 174 passed across experimental objective/dispatch/persistence,
  foundation contracts, persistence/unified persistence, growth, formula,
  survival and distributional tests. Run with `OPENBOOST_BACKEND=cpu
  NUMBA_NUM_THREADS=1 uv run --no-sync pytest <these test files> -n 0 -q`.
  Includes seed replay with plugin and builder RNG consumption and unchanged
  global NumPy RNG state. Production/changed-test lint and docs build passed
  (existing griffe documentation warnings remain).
- The combined callback run was interrupted during the old 500/1000-round
  GBDT early-stopping cases. Faulthandler showed repeated Python sample/tree
  prediction through `_fit_cpu -> predict -> _predict_standard_cpu`; reducing
  Numba threads did not remove that cost. These three long tests are not claimed
  as passed. Focused callback and new coefficient restoration checks substitute
  for this slice; no unrelated predictor optimization was made.

## P3 completion and isolated wheel evidence

- Focused legacy callbacks: 8 passed / 3 deselected; combined with the 174-case
  run, 182 relevant tests passed. The three deselected long early-stopping
  tests remain unverified in this slice, not a passing full CPU suite.
- Clean source `f414b8c` built wheel `openboost-1.0.0rc1-py3-none-any.whl`,
  SHA256 `5d0560682f7940252c15c9d5f8a67ccbe64849fdd8515aeaf600778c5595439c`.
- Isolated Python 3.12.12 CPU environment: NumPy 2.2.6, Numba 0.61.2,
  llvmlite 0.44.0, SciPy 1.16.3, joblib 1.5.3. Actual import came from temporary
  uv `site-packages`, with `python -I`; no training fixture module was imported.
  Three rounds of nonconstant coefficients reproduced both raw channels exactly.
- Failed installation attempts: offline cache lacked joblib; unpinned uv selected
  Python 3.14 / Numba 0.67 / llvmlite 0.49 and failed building against local LLVM
  20 (required 22). Pinning the development Numba 0.63.1 still required an x86
  macOS source build and failed in setuptools with `dry_run`. Public binary
  wheels for Numba 0.61.2 succeeded with `--no-build`. No broad fresh-install
  compatibility claim follows; dependency/platform packaging remains a follow-up.
- Reproducible harness: `tests/check_experimental_wheel_inference.py` (create
  mode imports deliberately unpicklable test plugins; verify mode imports only
  NumPy and the installed public OpenBoost API). From the repository, run:

```sh
OPENBOOST_BACKEND=cpu uv run --no-sync python -m tests.check_experimental_wheel_inference create /tmp/ob-inference
uv build --wheel
OPENBOOST_BACKEND=cpu uv run --isolated --no-project --python 3.12 --no-build \
  --with /absolute/path/to/dist/openboost-1.0.0rc1-py3-none-any.whl \
  --with numba==0.61.2 --with numpy==2.2.6 --with scipy==1.16.3 --with joblib==1.5.3 \
  python -I /absolute/path/to/tests/check_experimental_wheel_inference.py verify /tmp/ob-inference
```

- G1 is satisfied within the documented CPU/scalar-tree boundary. GPU extension
  primitives and dispatch are P4/P5; independent extension packages are P6.
  No external adoption or performance gain has been demonstrated by P3.
- Implementation commits: `50b3631` objective facade, `b2b3a7a` builder/schedule,
  `f414b8c` persistence/callback state. Subsequent changes only document evidence
  and add the standalone verification harness; library source is unchanged.
