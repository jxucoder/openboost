# Sprint 010: Finite compositions and F0.2 exit audit

Starting revision: `972f80a`. Status: complete. Fill A5/A7/A9/C4 reference compositions and audit F0.2 exit.

## Plan and acceptance

1. Immutable scalar ensembles retain base/terms/coefficients. Poisson raw excludes offsets;
   check two rounds and new exposure. Actually fit paid-count rate×Gamma severity and three parallel quantile models.
2. Connect ordered Normal updates to versioned proposals, train/validation caches and best restoration.
   Reject stale parents/NaN validation without state changes. Restore matching tree/coefficients/cache/step.
   Derive RNG from run ID/seed/logical step; retries cannot consume future keys.
3. Regression/lint and A1–A13/D1–D5 reference audit. Close only evidenced stages and identify F0.3/F1 follow-ups.

No serialization/public runtime or real quality/speed/author-cost evaluation; those remain F1/F0.3–F4.
Compositions may reuse references, never production imports or call memory restoration an artifact round trip.

## Results and verification

Nine new checks; 288 passed, no skips. Poisson raw remains log-rate over two rounds; new exposure
changes count mean only. Paid-count joins feed actual reference Poisson/Gamma fits; ClaimNb cannot
replace positive-payment counts. Three quantiles retain/reconstruct base, trees and coefficients.
Four ordered Normal commits across two rounds use versioned state. Base and nonzero best snapshots
restore all terms, both caches and logical step; fresh versions reject old proposals.
Reverse-direction rejection, non-finite validation, foreign runs, corrupted raw and broadcastable
row/shape errors leave original state intact. Step-derived keys survive retry/restoration; no device/global RNG state exists here.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

Local macOS/Python 3.12.12/NumPy 2.3.5; tests/lint/strict docs pass. Isolated reference compositions
run while every openboost import is blocked. See the [F0.2 ledger](f0-2-acceptance-ledger.md).

## Reflection: End mathematical preparation and advance the evaluable foundation

Independent mathematics, counterexamples and finite two-round compositions now cover A1–A13/D1–D5.
Close F0.2 preparation instead of indefinitely adding isolated fixtures and delaying the product.
No production conformance, persistence, real quality, author-cost or GPU gate closes with it.
Review corrected the nonzero-best fixture: initialization and updates must use the same metric
for meaningful comparisons. Explicit shape checks are also needed; allclose does not prevent broadcasting.

Next F0.3: real manifests/hashes, capability smoke, budgets, held-out tasks and judge, retaining
every required case. F1 builds public components checked against these oracles. Reference Track
is a finite Normal-state probe, not a production transaction protocol or disk checkpoint.

## Commits

- This slice: `test: integrate reference models and close v1 F0.2`.
