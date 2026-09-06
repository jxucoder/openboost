# Sprint 014: Adult official test and stratified splits

Starting revision: `f74df58`. Status: complete for A2 raw data preparation.
The user reaffirmed evaluation-first execution: retain F0.3 before F1, with no
premature switch to production GPU implementation.

## Plan and acceptance

1. Pin UCI archive and adult.data/adult.test hashes. Preserve official test, exclude
   fnlwgt from 13 features, map categorical `?` to None, strip exactly one test-label
   suffix dot, and use unit weights.
2. For seeds 0–4, use default_rng(seed), permute class 0 then class 1, take
   floor(0.8*n_class) for training and the remainder for validation. Sort returned
   source indices. Freeze source row IDs, hashes and class counts.
3. Replay real data, test without network, run regression/lint, document and commit.
   Smallest counterexample: changing fnlwgt cannot change X or sample weights.

## Results and verification

The archive contains 32,561 official training and 16,281 official test records.
All five test partitions retain identical row IDs. The adapter preserves missing
categories and source-qualified physical-line IDs; no category vocabulary is learned.
Duplicate predictor rows are counted in the artifact, not silently deleted or
interpreted as proof of repeated people. No person identifiers are available.

[The frozen record](../benchmarks/v1/datasets/adult.json) binds raw member hashes,
parsed records, five splits, source bytes and execution provenance. Real archive
replay matched. Sixteen tests cover exclusions/missingness, strict labels, category
separation, class proportions, invalid inputs, RNG isolation and archive identity.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.adult /tmp/openboost-v1-adult.zip --verify benchmarks/v1/datasets/adult.json
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

Local macOS/Python3.12.12/NumPy2.3.5. No models trained or test metrics inspected.
Numeric encoding, baseline capabilities, budgets and quality evaluation remain pending.

## Reflection

The fixed official test must not be recombined with training before splitting.
The numeric population weight column is not automatically a sample weight; removing
it from X is an explicit task contract. Repeated predictor values cannot establish
entity identity. The initial test failed because the adapter did not yet exist.
The user additionally requires English throughout the repository; existing prose
will be translated separately while preserving all specifications and evidence.

## Commits

- This slice: `data: freeze Adult official test and stratified splits`.
