# 2026-09-07: Normal loss changes need an arithmetic enclosure

## Context

The user approved local Sprint 092 after run 7 measured an actually worsening
candidate accepted by both device and original-row float64 loss subtraction.
All seven GPU invocations remain consumed. This slice concerns independent
mathematics before the public programmable comparison boundary.

## Decision or Result

The [bounded original-row prototype](../tests/v1/reference/normal_comparison.py)
uses the stable Normal difference identity with outward arithmetic throughout.
Its exponential enclosure uses a Taylor remainder and repeated doubling/squaring;
it does not assume a guaranteed platform-libm ULP bound. CUDA's published standard
function bounds explicitly come from nonexhaustive tests. The
[derivation, references and limitations](../v1-sprints/092-normal-comparison-mathematics.md)
state the conditional IEEE arithmetic assumptions and bounded exponent domain.

On the actual stored run-7 inputs, both recorded mean steps in both orders are
resolved as worsening. The rate-four enclosure is approximately
`[5.904685e-18, 5.905035e-18]`. The analytic `-2^-61` improvement remains resolved;
unchanged stored raw and unresolved cancellation remain separate outcomes.
Invalid rows cannot be hidden behind zero weight. No production correction or
device conformance is claimed.

## Changes

- Independent interval experiment and distinguishing tests; historical oracles
  and failed raw artifacts are unchanged.
- Sprint 092 approval, explicit numerical policy, supported domain and public
  operation prerequisites recorded in the active sprint and repository guidance.
- [All 383 historical cases](../v1-sprints/092-historical-case-mapping.json) map to
  236 unchanged operation requirements, 96 mapped transactions, 31 recipes and
  twenty installed-extension/inference requirements. Both historical failures
  retain their status and explicit predicate-supersession flags; none of the
  planned revised requirements is labelled implemented or passing.
- The [study harness](../benchmarks/v1/normal_comparison_study.py) retains 106
  numerical cases: twenty comparisons of captured GPU inputs, five analytic cases
  and 81 actual proposals from the frozen CPU oracle across ordinary/Fisher/damped
  and joint/ordered learners. No device trajectory is emulated.

## Verification

- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_normal_comparison_reference.py tests/v1/test_normal_acceptance_reference.py -n 0 -q --tb=short`.
- Result: **40 passed**, including all 25 new comparison tests and fifteen
  unchanged historical acceptance-oracle tests; production/new-support lint passes.
- New tests check both trace orders on training/validation, original-row Decimal
  agreement, exact-rational basic arithmetic, exponential endpoints, f32/f64 whole
  expressions, weight transformations and invalid/unsupported domains.
- Ruff check and formatting cover only the new support files. Full regression
  results and the complete reproducible case mapping are recorded below.
- CPU Python 3.12.12 on macOS; no CUDA run, emulator or host fallback.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_normal_comparison_study.py tests/v1/test_normal_comparison_reference.py -n 0 -q --tb=short`: **27 passed**.
- Full CPU regression after the cohort harness: **1505 passed, one Linux-only
  skip**. Production/changed support lint passes. MkDocs builds with the existing
  run-6 evidence link outside the documentation tree warning; no new warning.
- [Raw local study](../benchmarks/v1/evidence/normal-comparison-092/study.json)
  generated at clean `a5967b5`: **38 improvements, 59 worsening changes, seven
  unchanged and two unresolved**, across 106 declared comparisons. All 105
  available bounds contain the high-precision estimates. The unsupported
  exponent case has no manufactured estimate/bound. Source hashes and every
  numerical case are checked against the archived clean-source result.
- Closure verification: **66 focused numerical/archive tests pass**; the full
  CPU suite passes **1506 tests with one Linux-only skip** after adding archive
  reproducibility. Production/changed support lint and MkDocs pass; the existing
  documentation-tree link warning remains. No production source changed.

## Failed Attempts

- A stable expression with a standard library `exp` error budget would silently
  promote measured library accuracy to a guarantee. Use a derived bounded
  approximation instead; actual compiled implementation remains unverified.
- The 60/100-digit oracle agreement check fails its forty-relative-digit target
  for the `2^-52` scale step at a stationary point. Subtraction loses about 32
  digits. Retain the original estimates and add 160/220 digits; do not weaken the
  comparison policy or modify the frozen oracle.

## Risks and Follow-ups

- This is conditional mathematical evidence, not formally verified machine code.
  Polynomial and interval costs need measurement after public construction.
- Unsupported exponent/intermediate ranges are unresolved, not clipped or sent
  to the CPU from CUDA. Production Normal domain checks remain prerequisites.
- The complete 383-case mapping is published before 092-B. Public operations,
  distinct training/best/stopping anchors and a separately approved hardware
  freeze remain. The [slice reflection](../v1-sprints/092-normal-comparison-design.md#092-a-closure-and-reflection)
  keeps this addition objective-owned, with no expansion into general math
  libraries, CPU optimization, or unsupported algorithm-value claims.

## Commits

- `8d0e92f` — independently verified interval mathematics and numerical contract.
- `a5967b5` — complete historical mapping and numerical study harness.
- Clean-source evidence and 092-A reflection are archived in the closure commit.
