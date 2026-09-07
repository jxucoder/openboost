# 2026-09-07: Resident Normal operations before runtime integration

## Context

The numerical freeze `f75f501` precedes kernels. The user asked to continue across
verified slices without stopping. 090-B constructs the first Normal device
components while preserving the existing scalar path and all 212 old verifiers.

## Decision or Result

Normal preparation/base/geometry/loss are explicit public operations. Generic
diagonal direction and per-column least-squares fields can serve other algorithms.
ObjectiveOperations is an explicit callable dependency record; runtime integration
comes next. All new device behavior remains unverified on hardware.

## Changes

- `device_normal.py`: Normal schema, offset-aware normalized initialization,
  unweighted gradient/Fisher and weighted NLL. Per-row float64 intermediates must
  produce representable float32 geometry and positive scale/Fisher; no clipping.
- `device_objectives.py`: target/raw widths, K-column broadcast, objective
  dependency bundle, diagonal directions and once-weighted direction fields.
- `_device_kernels.py`: new kernels only; existing scalar kernel bodies unchanged.
- 23 collected real-CUDA checks and ten CPU import/floor checks. Existing input
  ownership and atomic scratch cleanup are reused.

## Verification

- First `test_device_normal_api.py` run failed at import before implementation.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_device_normal_api.py tests/v1/test_device_api.py tests/v1/test_normal_precision_reference.py -n 0 -q`
  — 40 passed.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' -q --tb=short`
  — 1392 passed, one Linux-only skip; macOS/Python 3.12.12. Local log:
  `/tmp/openboost-090-b-cpu.log` (not a committed benchmark artifact).
- `uv run --no-sync pytest tests/v1/test_device_normal_cuda.py --collect-only -n 0 -q`
  — 23 collected, zero hardware executions. Same cache environment as above.
- Ruff for production/new tests and `uv run --no-sync mkdocs build` pass. Staged
  diff inspected before commit. No GPU simulation substitutes for validation.

## Failed Attempts

- Initial code multiplied unnormalized weight by relative precision. Review
  found representable inputs (`weight=1e30`, offset log scale `-345`) whose
  product exceeds float64 even though normalized initialization is finite.
  Normalize first, preserve the same equation, and retain a hardware check.
- An early generic direction implementation allowed a zero metric when damped.
  CPU semantics require positive diagonals, including ordinary mode. Reject zero
  before output validation; two hardware cases cover that boundary.

## Risks and Follow-ups

CUDA compilation, numerical parity, real transfer counts and resource cleanup
are pending the eventual approved hardware run. Next 090-C generalizes the
existing runtime to mapped terms and K-column transactions; do not add a second
Normal trainer. Known near-tie structural ambiguity remains diagnostic, not a pass.
All hardware/upload allowances remain consumed; no remote action occurred.

## Commits

- `f75f501` — numerical preregistration.
- `ab6a37b` — 090-B resident operation construction.

## 090-C mapped runtime construction

DeviceRun accepts the explicit objective record, keeps K-column raw state and
snapshots tuples of mapped scalar trees. One tuple is one transaction; best-prefix
selection remains independent. `DeviceTerm` owns immutable mapping metadata and
`validate_terms` separates structural checks from numerical search. `map_update`
passes small maps as kernel arguments, preserving scalar proposal no-upload
checks. K=1 identity maps retain the existing scalar kernel. All shared terms
keep their independent reference counts through proposal/parent release.

Thirty-seven CPU API checks pass, including immutable maps and invalid metadata;
96 new real-hardware cases collect. Ninety cover the frozen three-round matrix;
six cover mapped lifetime/schema and partial failures in second-tree copy,
prediction, validation and resolution. Full CPU regression is **1400 passed,
one Linux-only skip** (`/tmp/openboost-090-c-cpu.log`), using the same command as
090-B. Ruff and documentation build pass; staged diff reviewed before commit.
No GPU execution occurred. Existing hardware test files remain unchanged.

This third local slice triggers a reflection recorded in Sprint 090. The shared
runtime now expresses Normal updates; the next step is actual joint/ordered
recipe composition and an installed D2 consumer. No host training fallback,
second private trainer or new hardware authority was introduced.

## 090-D recipe composition

`782c9c4` commits the shared runtime. The next slice adds Normal joint/forward/
reverse recipes and public `try_terms`. Numerical search keeps every scalar
coefficient, candidate training/validation score, decision and failure. State/term
schema is validated before retry logic. Caller-owned previous state is retained
by the search operation; the built-in recipe releases superseded states itself.
Stopping observes each outer sweep once, with no per-round arrays in history.

The 51 targeted API checks and full **1414 CPU tests** pass with one Linux-only
skip (`/tmp/openboost-090-d-cpu.log`, same regression command). Ruff/docs pass.
Thirty-one real-CUDA recipe cases collect; none executes locally. They cover
fixed/backtracking trajectories, zero budgets, ordered partial acceptance,
actual sixth-trial recovery, patience, second-learner failure and long-run/fresh
CPU inference. A released/forged state must fail before numerical search, which
motivated the public `validate_state` metadata check. No private optimizer or
CPU training fallback was added. Next: installed D2 and hardware package freeze.
