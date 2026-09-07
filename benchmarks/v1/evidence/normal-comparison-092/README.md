# Sprint 092-A independent Normal comparison evidence

This is local original-row mathematics at clean `a5967b5`, after the user approved
Sprint 092. It is not a production component, CUDA run, revised boosting
conformance, end-to-end performance result or author-benefit evaluation.
No GPU invocation or upload occurred. All seven device allowances remain consumed.

The raw [study.json](study.json) retains 106 comparisons, their exact inputs,
intervals, status/reason, high-precision differences, source hashes, CPU environment,
clean revision and command. The
[383-case historical mapping](../../../../v1-sprints/092-historical-case-mapping.json)
and [cohort specification](../../../../v1-sprints/092-comparison-cohorts.md) were
committed before this study. All planned revised cases remain unimplemented/unrun.
The original 381/383 and 383/385 device verdicts remain unchanged.

## Results

| Origin | Comparisons | Interpretation |
| --- | ---: | --- |
| Actual run-7 trials | 20 | Ten captured proposals, each on training and validation; original float32 bits remain in their hashed source traces. |
| Declared analytic cases | 5 | Tiny true improvement, cancellation, unchanged raw, clear worsening and unsupported exponent range. |
| Independent CPU proposals | 81 | Weighted/D2 original-row proposals across ordinary/Fisher/damped and joint/forward/reverse learners, including retries. These are not CUDA trajectories. |

Statuses: **38 improvement, 59 worsening, seven unchanged, two unresolved**. All
105 available enclosures contain the independent original-row high-precision
estimates. The remaining case deliberately lies outside the exponent support and
has no fabricated bound or estimate. One cancellation case needs 160/220-digit
oracle confirmation; its original 60/100-digit agreement failure is retained.

Both captured mean trials in each order are worsening. At coefficient four the
training interval is approximately `[5.904685e-18, 5.905035e-18]`, while the
historical device total-loss predicate accepted it. Its validation change is
negative: training acceptance and best-model selection require different inputs.
The exact `-2^-61` improvement remains resolved. The scale-only cancellation case
and unsupported range explicitly return unresolved.

## Reproduction and scope

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.normal_comparison_study study /tmp/openboost-092-comparison-study.json
OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/v1/test_normal_comparison_study.py tests/v1/test_normal_comparison_reference.py -n 0 -q --tb=short
```

At the recorded revision with a clean tree, numerical fields and source hashes
reproduce. Later command/revision/dirty metadata can differ; the archive test
compares all numerical cases and verifies each analysis source against the recorded
Git object. Inputs are synthetic or the previously committed diagnostic fixtures;
no dataset was downloaded. This small arithmetic study makes no timing claim.

The [derivation](../../../../v1-sprints/092-normal-comparison-mathematics.md) states
the IEEE rounding assumptions, Taylor remainder, bounded exponent range and
unresolved policy. It does not use measured libm accuracy as a universal guarantee.
Production operations and actual CUDA arithmetic/lowering/cost remain unverified.
The next local slice is 092-B; best/patience ownership and consumer migration follow
in 092-C, then a separately approved concrete device freeze in 092-D.

Closure verification: 66 focused numerical/archive tests pass. The full CPU suite
passes 1506 tests with one Linux-only skip. Lint passes and MkDocs builds with its
pre-existing link warning for the run-6 evidence outside the documentation tree.
