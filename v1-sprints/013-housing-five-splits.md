# Sprint 013: Five Housing splits for A1/A11

Starting revision: `dd2e9ad`. Status: data-hash/five-split subset complete; license unresolved.

## Plan and acceptance

1. Verify the cached archive against old hashes; independently implement column mapping,
   household ratios and target units. A hand-calculated row catches ratio/column mistakes.
2. Match combined X/y hash and old seeds 0–2; add3–4 and member/all-split hashes. Check completeness,
   disjointness, RNG isolation, invalid inputs and float32 overflow.
3. Real replay, regression/lint/docs and commit. A1/A11 share data but require separate quality results.

No model/GPU training. If source licensing is not verified, record unresolved rather than inventing a label.

## Results and acceptance

Cached archive matches fixed SHA256. The independent nine-column adapter computes AveRooms,
AveBedrms and AveOccup using households, preserving100,000 USD targets and little-endian float32.
The20640×8 combined X/y hash and all nine old split hashes match; six new hashes cover seeds 3–4.
[Freeze](../benchmarks/v1/datasets/housing.json) stores member hash, five splits, source and environment.
Each partition is12384/4128/4128 and internally complete/disjoint. Random splits do not establish geographic generalization.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python -m benchmarks.v1.housing build/foundation_data/cal_housing.tgz --verify benchmarks/v1/datasets/housing.json
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost benchmarks/v1 tests/v1 tests/conftest.py
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync mkdocs build --strict
```

Real replay matches; corrupted seed 4 makes CLI exit nonzero. Added20 tests; 380 passed, no skips.
Invalid households/counts/shapes/float32 overflow fail. Hand ratios, no cross-row learning and
unchanged global RNG pass. Local macOS/Python 3.12.12/NumPy 2.3.5. Raw data is not in Git;
no models or test metrics were run.

## Reflection

Historical records are reusable but have only three seeds, and the archive lacks license details.
Preserve hash semantics, verify independently and add two seeds. Public availability does not
establish a particular license. A1 and A11 need separate RMSE and NLL/CRPS results, not two source counts.
Initial missing-module collection failed; an early test-line semicolon was fixed by formatting.
Next Adult categories/official-test splits, then other required datasets. Licensing, capabilities,
budgets, held-out tasks and the full runner remain pending; F0.3 stays open.

## Commits

- This slice: `data: freeze A1 A11 housing with five splits`.
