# 2026-09-07: Normal device boundaries and independent fixtures

## Context

After run 5 passed all 212 bounded scalar CUDA cases, the user continued the
foundation work. Normal K=2 is the next structural probe under Sprint 079. The
existing device runtime fixes raw width at one and owns one tree per proposal;
copying it into another private trainer would not establish composability.

## Decision or Result

[Sprint 090](../v1-sprints/090-normal-device-construction.md) defines six slices:
independent math, resident Normal operations, mapped multi-term transactions,
joint/ordered loops, installed D2/fresh inference and a separately approved
hardware package. It preserves original P7, 069 accounting and all formal scope.
An explicit objective operation bundle separates the runtime from squared-only
validation; algorithm code chooses geometry, direction, learners and acceptance.

The 101 local checks include 90 three-round transaction comparisons using
original-row float64 math independent of production imports. They pass for
ordinary, natural and damped directions; fixed and actual bounded search;
joint/forward/reverse; weights/offsets/missing values; D2 minima and distinct
validation. This is designer development work, not independent author evidence.

Fisher belongs in direction construction. Trees regress those unweighted
directions using `G=-w*z, H=w`; adding independent cohort fields must not change
their objective-weight role. A joint proposal appends two terms but commits one
version. Ordered updates can reject mean then accept scale, with one outer stop
observation and separate best selection. Undamped natural mean directions cancel
precision, so reverse and joint updates need not differ mathematically.

## Changes

- `v1-sprints/090-normal-device-construction.md`: construction, ownership,
  numerical, installed-extension, evaluation and allowance boundaries.
- `tests/v1/reference/device_normal.py`: scalar-loop base/geometry/directions,
  original-row trees and complete small transaction trajectories.
- `tests/v1/test_device_normal_reference.py`: public CPU comparisons, analytic
  checks, actual rejected/invalid trials, partial rejection, and an import-blocked
  subprocess. Captures a separate unresolved numerical diagnostic.
- AGENTS/current sprint index/079: point the next local work at 090-B.

## Verification

- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_device_normal_reference.py -n 0 -q --tb=short`
  — 101 passed; first pre-construction run failed because the oracle was absent.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost/ tests/v1/reference/device_normal.py tests/v1/test_device_normal_reference.py`
  — passed.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short`
  — 1371 passed, one Linux-only skip; macOS/Python 3.12.12, 16 pytest workers.
- `UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build`
  — passed. `git diff --check` passed before staging; staged diff reviewed before commit.
- No production implementation changed. No new CUDA invocation/upload occurred;
  the earlier 212-device pass remains tied to `af026ef`.

## Failed Attempts

- Multiplying a NumPy float32 fixture weight by Python floats retained float32
  under NumPy 2 scalar-promotion rules. The independent base initially missed
  stationarity by about `5.17e-7`. Explicit Python-float row arithmetic fixed the
  oracle at its declared float64 precision; no tolerance was relaxed.
- Requiring every order to differ was mathematically wrong for an undamped
  natural mean update. Ordinary mode distinguishes all orders; undamped natural
  reverse/joint agrees to float64 rounding. Keep both assertions.
- The original two-feature weighted fixture produces almost equal histogram
  gains with different routing on a zero-weight row. Observed gains were
  `1.1156968959413` / `1.1156968959413003`, and depth-2 predictions differed by
  about `.00925`. The aggregates differ from summation order, unlike 089's exact
  swapped inputs. Retain `weighted_ties` and captured G/H as a diagnostic.
  Giving the extreme row positive mass alone still left deeper equivalent
  conditions. The final main fixture uses one feature/depth 1 with a separate
  zero-weight row; existing device fixtures remain unchanged. No parity pass is
  inferred for the ambiguous case and no score-selection epsilon was introduced.
- In the off-optimum scale fixture, the second candidate's scale is finite but
  its precision underflows. The actual accepted coefficient is .05, after both
  .2 and .1 fail. Float32 has different representability limits; freeze them
  explicitly before CUDA implementation.

## Risks and Follow-ups

- Implement 090-B only after freezing float32-domain cases/tolerances. Resolve
  the documented distinction among ambiguous mathematical ties, exact stored
  ties and well-separated winners before declaring device parity.
- Normal CUDA operations, mapped runtime, installed D2, fresh exported inference,
  device ownership/retention/cost and original P7 reproduction remain unverified.
- All five hardware allowances are consumed. Prepare the complete bounded
  package before requesting a new upload/invocation allowance. No agents launched.
- Normal is one required structural probe, not a substitute for the other
  algorithm/application families or formal author/value/adoption evidence.

## Commits

- Parent `48a1386` — run-5 evidence and scoped retrospective.
- `05b7c66` — 090-A design/oracle slice.

## 090-B numerical preregistration

Before adding kernels, 11 new CPU reference checks pass for explicit float32
geometry support and the sixth-trial recovery case. The numerical amendment in
090 freezes storage, float64 intermediate arithmetic, comparison tolerances and
the separate near-tie diagnostic. It does not claim that float64 and float32 have
identical overflow boundaries. No production code or earlier device fixture was
changed in this freeze. Verify with `uv run --no-sync pytest
tests/v1/test_normal_precision_reference.py -n 0 -q` (11 passed), using the same
`UV_CACHE_DIR` as above. Ruff passes for the two new Python files. Next implement
the declared resident operations; no new GPU/upload authority is inferred.
