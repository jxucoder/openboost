# Sprint 003: Independent training transforms and classification

Starting revision: `cdce7d8`. Status: this sprint complete; F0.2 ongoing.
Scope: B01/F0.2, mathematical/data semantics subset of A1–A3/C1.

## Plan

1. Write failing examples for linear quantiles, minimum-value cuts, missing/unknown values,
   nonordinal categories and label mapping.
2. Implement independent transforms and binary/softmax mathematics; check hand values,
   finite differences, weights and two-round updates.
3. Record scope/results/reflection, run current v1 regression/lint and commit separately.

## Acceptance and boundaries

- Fit cuts/dictionaries on training only; validation cannot mutate state. Out-of-range values
  use endpoint bins; reject infinity.
- Immutable column records with explicit missingness; stable dictionary one-vs-rest categories,
  with unknowns following missing routing.
- Label mapping preserves probability-column meaning; unknown/missing labels fail. Reject
  single-class binary training and require explicit initialization clipping.
- Stable binary loss/g/h; name exact softmax Hessian separately from tree diagonal upper bound.
- Apply weights only in loss aggregation/tree statistics, not prematurely in derivatives.
  Round two uses every output direction from the new raw state.
- PreparedData identity, row-ID binding, persistence, full categorical growth and all F0.2
  remain incomplete. Production APIs, real data, CUDA and E-gates are not implemented.

## Results and evaluation

**Bounded sprint complete; F0.2 ongoing.** Added data.py/classification.py, 40 tests and
extended isolated execution blocking every `openboost` import.

- Numeric hand values: `[0,2,4,6]` gives cuts 1.5/3/4.5. Repeated values retain a minimum-value
  cut. Cases cover constant/all-missing columns, one bin, endpoints, NaN/infinity and extreme finite interpolation.
- Categories: stable training dictionary, unknown-as-missing and both missing routes.
  Middle category m has one-vs-rest gain 64/15, strictly better than any ordinal threshold.
- Binary: at zero raw, g=±1/2 and h=1/4. Nonunit weights equal replicated rows; ±1000 loss
  stays finite, and correctly directed raw=±100 retains a small nonzero loss without cancellation.
  Initialization clipping must be explicitly supplied and representable.
- Softmax: hand-check K=3 exact Hessian and 2p(1-p) bound; finite differences check g/H.
  Test shifts/class permutations, weights, invalid labels and range beyond float64.
- Two rounds: binary first leaves ±2/3; second leaves independently use p=sigmoid(-1/15).
  Weighted softmax root first vector is (-3/11,0,3/11); all second directions use complete raw1.
  A split multiclass tree is equivariant to class order over two rounds; train/validation raw match.

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync pytest tests/ -n 0 -q
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync ruff check src/openboost tests/v1 tests/conftest.py
```

**95 passed, no skips**; Ruff passed. Local macOS CPU/Python 3.12.12/NumPy 2.3.5/pytest9.0.2.
Before implementation, collection had two missing-module errors. No CUDA, real-data baseline,
persistence or formal E-gate ran; 95 tests do not mean 95 product features.

## Reflection

Binning/categories define candidates; classification output axes define geometry. These cannot
be hidden in a trainer. Minimum-value cuts, middle-category counterexamples and same-snapshot
softmax round two expose different errors. Preserve independent transform/output-schema/geometry
boundaries without introducing a universal trainer or production compatibility layer.
Immutable column state is not full PreparedData identity or typed Problem; category routing is
not complete categorical growth. Next prioritize ranking/quantile/vector leaf requirements,
then remaining binding, categorical growth, positive/AFT, Normal/Formula and state/run work.
All A1–A13 remain required; this provides future public-component conformance oracles.

## Commits

- This slice: `test: add independent data and classification references for v1`.
