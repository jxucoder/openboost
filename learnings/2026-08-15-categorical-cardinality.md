# 2026-08-15: Categorical Tree Cardinality

## Context

Categorical values are encoded into `uint8`, so `BinnedArray` can represent up
to 254 non-missing categories. Tree nodes, however, represent the categories
sent left with one `uint64` bitset. The split search previously evaluated all
categories but silently omitted category codes 64 and above from that set.
Reported gain and actual routing could therefore disagree.

## Decision or Result

Keep the encoding limit at 254 because non-tree consumers can use those bins,
but reject categorical tree training above 64 categories in the shared split
dispatch before either the CPU or CUDA implementation runs. This preserves the
useful `BinnedArray` capability while preventing a tree from silently learning
an unrepresentable split.

Supporting more than 64 categories correctly requires a multiword bitset (or a
different category-set representation) across split search, partitioning, CPU
prediction, CUDA prediction, tree storage, and persistence. It is a feature,
not a safe one-line limit increase.

## Changes

- `src/openboost/_core/_split.py`: require category counts when categorical
  features are present and fail before dispatch when any count exceeds 64.
- `src/openboost/_array.py`: document the distinction between the 254-category
  encoding limit and the 64-category tree-split limit.
- `tests/test_categorical.py`: prove 65 categories can be binned but cannot be
  passed into categorical tree training.

## Verification

- Before the guard, the new 65-category training case did not raise.
- `uv run pytest -q tests/test_categorical.py tests/test_growth.py`: 46 passed.
- A focused 64-category probe separated category 63 correctly.
- `uv run ruff check src/openboost/_array.py src/openboost/_core/_split.py`:
  passed.
- `git diff --check`: passed.

## Failed Attempts

- The first design rejected more than 64 categories inside `ob.array()`. That
  conflated safe bin encoding with tree routing and would unnecessarily block
  consumers such as GAM. The guard was moved to the shared tree split path.
- Linting the entire legacy categorical test file surfaced pre-existing style
  issues unrelated to this change. The source files were used as the scoped
  lint gate; the complete behavior files were still executed with pytest.

## Risks and Follow-ups

- Multiword bitsets are required before advertising native high-cardinality
  categorical tree support.
- Real GPU verification should include category code 63 and CPU/CUDA parity;
  this machine has no CUDA environment.

## Commits

- `356103b` — `fix: reject unrepresentable categorical tree splits`
