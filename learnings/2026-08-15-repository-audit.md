# 2026-08-15: Repository Audit and Product Focus

## Context

A parallel code, benchmark, release, and ecosystem audit evaluated whether
OpenBoost had a credible path to a niche comparable in clarity—not impact—to
XGBoost. The audit was read-only and used local tests plus current primary
sources for competing libraries and publication routes.

## Decision or Result

OpenBoost has a substantive CPU tree core and a strong distributional subsystem,
but it is not yet a trustworthy general-purpose boosting library. The product
focus is now **calibration-first distributional boosting for tabular risk**:
NaturalBoost, exposure-aware count/severity models, proper scoring, calibration,
custom distributions, and verified single-GPU acceleration.

Generic GBDT, GAM, DART, linear leaves, Ray, multi-GPU, out-of-core, GOSS, and
train-many must not share equal product priority. GPU remains strategically
important, but the next milestone is one correct and evidenced NaturalBoost CUDA
path—not broader unverified GPU surface area.

## Evidence

- CPU suite at audit time: 721 passed, 32 skipped, 3 deselected.
- Total coverage: 53%; `_models/_distributional.py` 95%, distributions 79%,
  CUDA backend 0%, multi-GPU 16%, distributed tree 17%.
- The only committed third-party artifact was a three-dataset, one-seed CPU
  NaturalBoost/NGBoost comparison showing approximate parity, not dominance.
- The strongest external validation opportunity was ScoringBench, which accepts
  probabilistic model wrappers and publishes proper-scoring leaderboards.

## Release-Blocking Findings

- Categorical persistence used field names different from `TreeStructure`, so a
  save/load round trip could change predictions.
- Categorical binning accepted up to 254 values while routing used one 64-bit
  bitset.
- GPU GAM training dropped the base score after its first prediction update.
- Ray/multi-GPU workers initialized predictions inconsistently with final model
  inference, and multi-GPU child histograms were approximate.
- The documented memmap out-of-core example passed a feature-major array to a
  sample-major high-level API; `batch_size` was not connected to model training.
- The performance CI baseline was absent and regenerated on fresh runners, so
  the check could succeed without detecting regressions.

These findings must be re-verified against current code before fixing; this
entry records the audit state, not permanent truth.

## Product and Evidence Gates

1. Remove silent correctness failures.
2. Establish deterministic CPU reference behavior.
3. Verify end-to-end CPU/CUDA NaturalBoost parity.
4. Submit full-suite ScoringBench results with raw artifacts.
5. Add a real exposure-aware insurance case study such as freMTPL2.
6. Seek external users and contributions before JOSS/JMLR software submission.

## Risks and Follow-ups

- Distributional boosting mostly models aleatoric uncertainty; it does not by
  itself solve epistemic/OOD uncertainty.
- PGBM, XGBoostLSS, LightGBMLSS, NGBoost, CatBoost uncertainty, and Py-Boost
  already occupy adjacent positions. GPU probabilistic boosting alone is not a
  unique claim.
- A benchmark is allowed to falsify the product hypothesis. Quality regressions
  cannot be traded for speed without an explicit decision metric.
