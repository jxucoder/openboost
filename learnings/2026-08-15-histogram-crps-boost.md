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
- The complete non-GPU/non-benchmark CPU suite passed with 764 tests and 32
  expected skips. Public documentation marks the estimator CPU/numeric only,
  explains its finite-support risk, and makes no benchmark-win claim.
- The normal MkDocs build passes. Strict mode still aborts on 29 pre-existing
  Griffe warnings in older callbacks, distributions, losses, models, array,
  tree, and importance docstrings; none originate from `HistogramBoost`.
- `evaluate_crps_candidate.py` turns the preregistered thresholds into a
  fail-closed machine-readable decision: exact rows, one globally selected
  strong baseline, paired fold wins, CRPS ratio, coverage-error improvement,
  interval-score ratio, and RMSE ratio. This evaluator was committed before
  observing the new development outcome.

## Failed Attempts

- The first preregistered Actions run (`31930232251`) failed before dataset
  download because the launcher resolved a relative frozen-registry path after
  changing into its artifact directory. Registry inputs are now resolved
  before that directory change and covered by a regression test. The failed
  run observed no benchmark outcome and remains part of the audit trail.
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
- Public `sample_weight` still has a release blocker: rows with zero weight are
  removed from gradient aggregation but not from feature binning or target
  support. A zero-weight feature or target outlier can therefore change the
  fitted model. Delete zero-weight rows before both binning steps and add
  deletion-equivalence tests.
- The V2 workflow verifies result completeness and provenance but does not yet
  execute the acceptance evaluator in CI. The confirmation path is also not
  phase-bound. Keep `537_houses` locked until both are machine-enforced.

## First preregistered development result

The frozen `197_cpu_act` run completed 15/15 rows from clean source with the
preregistered dataset hash, but the candidate failed the development gate.
CatBoost was the strong baseline. HistogramBoost was 2.111% worse on mean CRPS
and won only two of five folds; its 98.23% coverage produced 8.23 percentage
points of absolute 90% coverage error. The interval-score and RMSE guardrails
passed. The confirmation dataset therefore remains locked.

HistogramBoost did beat native XGBoost quantile on all five folds, with 16.41%
lower mean CRPS, 24.93% lower 90% interval score, and 29.94% lower RMSE. This is
a one-dataset development signal, not an overall win. The candidate was badly
over-dispersed: mean sharpness 6.48 versus CatBoost's 1.56, while RMSE was only
0.34% worse. Diagnose retained tail mass on this consumed dataset before
changing capacity or touching confirmation data.

The follow-up audit found two semantic problems for V2. First, V1 trained a
midpoint ranked-probability approximation while ScoringBench evaluated the
exact piecewise-uniform histogram CRPS; the within-bin term is not a constant.
Second, `base_smoothing=1` added one pseudocount per bin, so prior strength grew
with output resolution. V2 uses the exact energy-form CRPS gradient and PSD
simplex-tangent curvature, and treats `base_smoothing` as a total Dirichlet
concentration spread evenly across bins. Both changes have analytic and
finite-difference tests; neither is yet benchmark evidence.

The training audit also showed that one absolute `reg_lambda` controlled both
split structure and leaf updates. Lowering it from 1 to 0.1 on a consumed fold
reduced sharpness strongly but changed the learned structure enough to worsen
RMSE. V2 therefore adds an optional `leaf_reg_lambda`: split regularization
stays at 1, while leaf shrinkage can be studied independently. `None` preserves
the original coupled behavior.

On consumed `197_cpu_act` fold 0, the exact-objective model at 100 rounds
improved CRPS from 1.3614 to 1.3187 and RMSE from 2.8446 to 2.7647, but official
90% coverage remained 97.67%. A diagnostic temperature sweep found that 0.7
improved CRPS again to 1.2957, RMSE to 2.7271, coverage to 94.0%, and interval
score to 10.9102. Because that temperature used an already-consumed outer fold,
it cannot be frozen directly. The ScoringBench wrapper now supports selecting
temperature by exact CRPS on an inner training-only split, then refitting the
base model on the complete outer training fold. Its default grid remains
`(1.0,)`, so V1 behavior does not silently change.

The inner split selected temperature 0.85 on consumed fold 0. Outer metrics
were CRPS 1.2986, RMSE 2.7148, coverage 95.67%, and interval score 11.7174.
Exact within-bin coverage was 93.17%, confirming that the remaining official
coverage excess came partly from whole-bin interval envelopes. The distribution
output now supports density-preserving subdivision, and the wrapper can refine
50 training bins to 100 evaluation bins. Subdivision leaves exact CRPS, mean,
and variance unchanged; it only reduces evaluator quantile-grid error.

## Frozen V2 development result

The frozen V2 Actions run `31932958804` completed 15/15 rows from clean
OpenBoost source `39bdb63` and pinned ScoringBench source `a938a667`. The
downloaded artifact had GitHub digest
`sha256:572672e818cf60a295826f7b057bad0269f5121f5bfc31a258188eea12234adc`,
and the compressed `197_cpu_act` input matched its preregistered SHA-256. The
fail-closed evaluator returned `development_pass=true`.

HistogramBoost V2 achieved mean CRPS 1.287204. That was 2.943% lower than
CatBoost MultiQuantile's 1.326236, with four strict fold wins out of five, and
20.550% lower than native XGBoost quantile's 1.620136, with five of five fold
wins. V2 also lowered RMSE by 1.200% relative to CatBoost and by 31.017%
relative to XGBoost. Its official 90% interval score was 11.281116 versus
13.329529 and 17.979673, respectively.

Official 90% coverage was 94.80%, so the frozen absolute-error guardrail passed
at 4.80 percentage points. This value is not a standalone calibration claim:
ScoringBench uses whole-bin interval envelopes and V2's lossless two-way
subdivision reduces that representation error without changing the density,
physical CRPS, mean, or continuous variance. Exact interpolated multi-level
coverage should become the future calibration gate.

The quality improvement did not make the CPU implementation fast. Mean fit
time, including the inner calibration fit, was 161.41 seconds per fold: 1.145x
CatBoost and 10.907x XGBoost in this one four-core Actions run. This converts
GPU vector histograms from a speculative feature into a concrete systems
target: preserve the frozen quality behavior while reducing the ten-fold CPU
gap to native XGBoost.

This remains consumed one-dataset development evidence. It does not establish
an overall ScoringBench, full-suite, or SOTA win. An independent audit also
found that the current gate mixes CatBoost with the XGBoost-family goal, omits
XGBoostLSS from the V2 run, and is not executed inside the workflow. The next
protocol slice must separate the strict XGBoost-family win, CatBoost
competitiveness, calibration/RMSE guardrails, and confirmation decision before
touching `537_houses`.

## Commits

- `b9db276` — `feat: add histogram CRPS boosting`
- `3cbd763` — `bench: automate CRPS candidate acceptance`
- `8ddd276` — `fix: align histogram training with continuous CRPS`
- `35ae55d` — `feat: separate vector split and leaf regularization`
- `ec810fd` — `feat: select histogram temperature on inner validation`
- `5f376e4` — `feat: refine histogram evaluation grids losslessly`
- `39bdb63` — `bench: freeze HistogramBoost V2 protocol`
