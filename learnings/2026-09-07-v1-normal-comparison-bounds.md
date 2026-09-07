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

## Verification

- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_normal_comparison_reference.py tests/v1/test_normal_acceptance_reference.py -n 0 -q --tb=short`.
- Result: **40 passed**, including all 25 new comparison tests and fifteen
  unchanged historical acceptance-oracle tests; production/new-support lint passes.
- New tests check both trace orders on training/validation, original-row Decimal
  agreement, exact-rational basic arithmetic, exponential endpoints, f32/f64 whole
  expressions, weight transformations and invalid/unsupported domains.
- Ruff check and formatting cover only the new support files. Full regression
  results and the reproducible case mapping follow before 092-A closes.
- CPU Python 3.12.12 on macOS; no CUDA run, emulator or host fallback.

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
- Publish the complete 383-case mapping before 092-B. Public operations, distinct
  training/best/stopping anchors and a separately approved hardware freeze remain.

## Commits

- This independently verified mathematics slice; concrete SHA linked at closure.
