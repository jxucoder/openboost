# 2026-09-05: Foundation P2 correctness

## Context

P2 starts with the suspected weighted constant-Hessian optimization conflict.
The unified objective weights both gradients and Hessians; the native hint
previously depended only on the unweighted objective's unit-Hessian property.
P2 also requires explicit capabilities, visible fallback and scoped seeds.

## Decision or Result

The initial seven host regressions failed: same-name custom distribution
misclassification, swallowed kernel errors, GPU sampling preflight (two
parameters), failed-first-step fitted state and missing model seed support
(two sampling parameters). They pass after the changes below.

A host interception test verifies that uniform and nonuniform weights now
select const_hess=0, while absent weights retain const_hess=1. This verifies
policy, **not actual weighted CUDA math**. The real-device regression harness
is committed, but its execution was rejected by automatic approval review.
No GPU failure or success was measured in this P2 slice; P2 remains incomplete.

## Changes

- `benchmarks/foundation`: add an isolated correctness suite alongside smoke.
  Require all smoke cases plus fixed-bin weighted Newton and Normal/Poisson.
- `tests/foundation/test_correctness.py`: include zero and nonuniform weights,
  inspect the native histogram hint and check analytic Newton predictions;
  compare CPU/device distribution gradients, splits, raw scores and NLL.
- `_trainer.py`: disable the unit-Hessian hint for weighted fits; validate
  sampling before binning; reject GPU sampling until verified; warn about host
  objectives and generic tree fallback. Restore trainer-owned state on errors.
- `_objectives.py`: only exact built-in distribution types select built-in
  kernels; let runtime and compilation errors propagate without a host retry.
- NaturalBoost/DistributionalGBDT, FormulaBoost and WeibullAFT now accept
  `random_state`. The unified trainer creates one local generator per fit,
  shared by row and column sampling. Core callers without a generator retain
  their legacy behavior; this is not a global legacy-RNG migration.
- Document current execution boundaries and preserve unweighted initialization
  semantics. No new performance or quality claims.

## Verification

All commands use `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`.

- `OPENBOOST_BACKEND=cpu uv run --no-sync pytest
  tests/test_foundation_contracts.py -n 0 -q`: initial **7 failed**, then
  expanded suite **12 passed**, including native-hint interception, visible
  host fallback, global RNG isolation and seed persistence/refit round trip.
- `OPENBOOST_BACKEND=cpu uv run --no-sync pytest
  tests/test_foundation_contracts.py tests/test_foundation_runner.py
  tests/test_distributional.py tests/test_distribution_gradients.py
  tests/test_formula.py tests/test_survival.py tests/test_growth.py
  tests/test_unified_persistence.py -n 0 -q`: **171 passed, 2 skipped** in
  39.62 seconds. The two skips require optional JAX; no GPU success is inferred.
  The additional seed persistence case was added and verified afterward.
- `uv run --no-sync ruff check src/openboost benchmarks/foundation
  tests/foundation tests/test_foundation_contracts.py
  tests/test_foundation_runner.py`: passed.
- `uv run --no-sync mkdocs build`: passed with existing griffe warnings.
- `uv build --offline`: wheel and source distribution built successfully.

## Failed Attempts

- Automatic approval review rejected the combined clean-wheel prepare / Modal
  command because it considered uploading this specific source-derived wheel
  and test bundle insufficiently explicitly authorized. No remote job ran.
  Do not reroute or indirectly perform the upload. Request explicit approval
  for the allowlisted bundle after completing local work.
- Lint found two nested context managers in new tests; combined them and reran.

## Risks and Follow-ups

- Run the committed pre-fix harness at `502f37f` and the fixed source on real
  T4 hardware, retaining failures and successes with clean source hashes.
- Complete GPU fallback/error checks, callback/eval transfer and cross-device
  persistence verification. The host tests alone do not establish CUDA parity.
- Freeze the real-data hashes, splits/seeds and cold/warm baseline only after
  those correctness checks pass. Do not advance to P3 as if P2 were complete.
- Rollback covers the trainer's assigned state. Arbitrary callback side effects
  and facade state modified before entering the trainer are outside that scope.

## Commits

- `1669974` — previous passing P1 evidence.
- `502f37f` — pre-fix weighted CUDA regression harness; no GPU results yet.
