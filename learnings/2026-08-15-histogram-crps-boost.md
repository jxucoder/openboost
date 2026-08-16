# 2026-08-15: Histogram CRPS Boosting

## Context

The corrected Gaussian NaturalBoost model reached parity with Gaussian
XGBoostLSS on the consumed `1028_SWD` ScoringBench dataset but remained 2.80%
behind native XGBoost multi-quantile on mean CRPS. CRPS training, early
stopping, scale calibration, Student-t, empirical residual shapes, and
independent quantile trees did not close that gap. The remaining limitation was
distribution shape, not only Gaussian scale optimization.

## Decision or Result

Add a separate non-parametric estimator, `HistogramBoost`, rather than forcing
many histogram logits through NaturalBoost's one-tree-per-parameter loop. Each
round learns one shared routing structure with a vector of logit updates in
every leaf. Softmax probabilities imply a valid monotone CDF, so quantile
crossing is impossible.

The training loss is a discretized CRPS (ranked probability score) on an
ordered target grid. Its curvature is the positive-semidefinite diagonal of the
Gauss--Newton matrix, `2 * diag(J.T W J)`. An earlier prototype that took the
absolute value of an indefinite exact Hessian diagonal is mathematically
invalid and was not carried into OpenBoost.

The frozen development defaults are 50 distribution bins, 100 shared-vector
trees, learning rate 0.05, depth 6, and curvature scale 1. These defaults were
chosen using only the already-consumed `1028_SWD` diagnostics. They must not be
tuned again before the preregistered `197_cpu_act` run.

## Changes

- `src/openboost/_core/_vector_tree.py`: CPU-only level-wise vector histogram
  tree with shared splits, vector Newton leaves, missing-value routing, and
  output-count-invariant gain/child-weight aggregation.
- `src/openboost/_models/_histogram_boost.py`: sklearn-cloneable
  `HistogramBoost`, train-only target support, empirical smoothed base logits,
  PSD CRPS gradients/curvature, sample weights, and full histogram distribution
  output (`mean`, `variance`, `quantile`, `interval`, and `sample`).
- Persistence reconstructs `VectorLeaves` with its output dimension and can
  auto-load `HistogramBoost`.
- Shared sample-weight validation now rejects all non-finite values.
- The model is exported from the public package, while the vector-tree builder
  remains internal until its CPU/CUDA contract is complete.

## Verification

- Focused model, persistence, growth, and validation suite: 82 passed.
- Ruff lint passed for every changed production/test file; the three new files
  pass Ruff formatting.
- Finite differences validate the CRPS logit gradient.
- An explicit Jacobian validates the PSD Gauss--Newton diagonal.
- Brute-force split/leaf checks cover vector gain and missing routing.
- Fit/predict tests cover monotone CDFs, train-only support, moments, quantiles,
  deterministic sampling, sample weights, sklearn cloning, and persistence.
- A local, non-artifact `1028_SWD` fold-0 smoke produced CRPS 0.327624 versus
  0.334891 for the frozen native XGBoost quantile row. This is consumed-data
  implementation evidence only and cannot support a product claim.
- The ScoringBench adapter exposes the frozen model as
  `openboost_histogram_cpu` and passes the model's regular PMF grid through the
  benchmark's native-grid path. The three wrapper contracts and 13 provenance
  tests pass against the pinned local ScoringBench checkout.

## Failed Attempts

- Absolute-value exact Hessian: improved the prototype but has no valid PSD or
  majorization interpretation; reject it.
- Gaussian residual-shape calibration and independent scalar quantile models:
  remained materially behind the native quantile baseline; do not productize
  them as the winning path.
- Reusing NaturalBoost's parameter loop: would fit 50 independent structures
  per round and lose the shared-tree scaling and non-crossing design.

## Risks and Follow-ups

- The implementation is intentionally CPU/numeric only. CUDA histograms and
  prediction kernels are scalar today; a GPU path needs output tiling and exact
  CPU/CUDA parity before it is enabled.
- Categorical splits, callbacks, evaluation sets, early stopping, subsampling,
  and column sampling are not implemented.
- Fixed train-range support can clip held-out extremes. The preregistered run
  must publish failures and all interval/calibration guardrails.
- Benchmark next on `197_cpu_act` using protocol `crps_distribution_v1`. Freeze
  one candidate before explicitly unlocking `537_houses`.

## Commits

- Pending — `feat: add histogram CRPS boosting`
