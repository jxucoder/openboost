# 2026-09-06: Categorical equality shares numeric growth operations

## Context

All three growth policies existed but interpreted conditions as numeric thresholds.
B07 requires typed category preparation, equality splits and persistent missing/
unseen routing through the same foundation rather than an alternate trainer.

## Decision or Result

MixedData owns tuple-backed mixed feature rows and explicit feature kinds. Binning
fits numeric cuts or sorted homogeneous string/integer dictionaries on training
inputs only. Categorical candidates select one value versus the rest, with both
missing routes. Unknown values follow missing routes. Histogram counts/fields,
choice, growth and transactions remain shared.

Binning/Tree replace NumericBinning/NumericTree with no compatibility aliases.
The tree format is openboost-tree-v2, including typed dictionaries; old numeric
formats fail loading. NumericData remains a numeric-only input. MixedData.values
is a detached array export, not a writable view of owned state.

## Changes

- data/binning: mixed input, dictionary validation, explicit schema kinds and
  feature-major codes/missing masks with transformer identities.
- ops/tree: equality candidate statistics and routing, persistent dictionary
  conditions, schema and category-index validation across all three policies.
- artifacts/recipes: mixed input through existing scalar recipes and mapped terms.
- Tests/docs migrate public names while preserving independent numeric references
  and prior mathematical/state checks. Historical evidence records remain intact.

## Verification

- Initial mixed-input test failed before implementation with missing MixedData.
- `OPENBOOST_BACKEND=cpu UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run
  --no-sync pytest tests/ -m 'not gpu and not benchmark' --tb=short -q`:
  615 passed, including 12 categorical cases. Python 3.12.12, NumPy 2.3.5, macOS.
- All three mixed-feature policy topologies/leaf values/predictions agree with
  independent exhaustive raw-row reference under nonuniform/zero weights.
- String/integer/all-missing dictionaries agree with independent CategoryMap;
  both equality missing routes conserve rows and weighted candidate sums.
- Fresh-process squared ensemble persistence preserves numeric/category/unseen/
  missing predictions. All-missing and missing-only categorical cases work.
- Invalid dictionaries, empty referenced dictionaries, foreign feature kinds,
  transformer identities and invalid/mixed category token types are rejected.
- Ruff, strict MkDocs and offline build pass. Every code example across seven
  public documentation pages passes under Python -I from an isolated installed
  wheel outside the checkout.
  Wheel SHA256: 4d632759ee2dd9293419e3ede66d6c4d7454d9228983c15f86d504084463ef94.

## Failed Attempts

An unused corruption-case label needed a lint rename. Review found the all-missing
histogram's placeholder bin must not be accepted as a real category condition;
Tree validates against dictionary length rather than padded histogram capacity.
No independent reference or mathematical acceptance threshold changed.

## Risks and Follow-ups

This is one-category-versus-rest splitting, not subset search or ordered target
statistics. MixedData exports object-array copies; no throughput or memory claim.
Inference artifacts are raw models, not resume checkpoints or calibrated output
schemas. Complete squared composition is verified here; accepting mixed inputs in
Normal/Formula does not establish their real-data categorical quality.

Next: B08 class schema, classification and vector-leaf/mapping probes. Ordered
updates, specialized leaf/objective families, CUDA, agent/adoption studies and
all required real application evaluations remain necessary. F0.3 and complete
F1/v1 acceptance remain open. No push or upstream-library parity claim.

## Commits

- This slice: feat: add categorical equality splits and mixed-feature artifacts.
