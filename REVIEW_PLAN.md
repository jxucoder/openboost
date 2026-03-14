# OpenBoost Code Review Plan

This document outlines the areas to review before the 1.0.0 release. It is organized by priority and category.

---

## 1. Core Algorithm Correctness

### 1.1 Tree Building (`src/openboost/_core/`)
- **`_tree.py`** (1328 lines): Verify `fit_tree()`, `fit_tree_gpu_native()`, and `fit_tree_symmetric()` produce correct splits and leaf values across edge cases (single-feature data, constant targets, all-missing columns).
- **`_primitives.py`** (893 lines): Audit histogram aggregation, split finding, and sample partitioning for off-by-one errors, integer overflow on large datasets, and correct handling of bin 255 (missing).
- **`_growth.py`** (1039 lines): Confirm that `LevelWiseGrowth`, `LeafWiseGrowth`, and `SymmetricGrowth` respect `max_depth`, `min_samples_leaf`, and `min_gain_to_split` constraints consistently.
- **`_histogram.py` / `_split.py` / `_predict.py`**: Review for consistency with the primitives they wrap.

### 1.2 CPU/GPU Parity (`src/openboost/_backends/`)
- **`_cpu.py`** (680 lines) vs **`_cuda.py`** (3465 lines): Verify that CPU (Numba) and CUDA (CuPy) backends produce numerically equivalent results. Pay special attention to floating-point reduction order, atomic operations, and thread-safety in CUDA kernels.
- Check that CUDA kernel launch configurations (block/grid sizes) handle non-power-of-2 dataset sizes correctly.

### 1.3 Loss Functions (`_loss.py`, 903 lines)
- Validate gradient and Hessian formulas for all 50+ loss functions against known references.
- Check numerical stability: log(0), division by zero, extreme predictions.
- Verify `get_loss_function()` dispatch returns correct implementations.

### 1.4 Distributions (`_distributions.py`, 1745 lines)
- Validate natural gradient computations (Fisher information matrix) for all 8 distributions.
- Check log-likelihood, CDF, and sampling methods against scipy reference implementations.
- Review `CustomDistribution` and `create_custom_distribution()` for correct JAX autodiff integration.

---

## 2. Model-Level Review

### 2.1 GradientBoosting (`_models/_boosting.py`, 1189 lines)
- Review the training loop: gradient computation, tree fitting, prediction update, learning rate application.
- Verify early stopping logic with validation set correctly tracks best iteration.
- Check GOSS sampling integration does not introduce bias.

### 2.2 NaturalBoost (`_models/_distributional.py`, 532 lines)
- Confirm multi-parameter boosting (e.g., mean + variance for Normal) updates parameters correctly.
- Verify that all 8 distribution variants (Normal, LogNormal, Gamma, Poisson, StudentT, Tweedie, NegativeBinomial) initialize and constrain parameters properly.

### 2.3 OpenBoostGAM (`_models/_gam.py`, 395 lines)
- Review single-feature tree constraint enforcement.
- Verify feature interaction detection and shape function extraction.

### 2.4 DART (`_models/_dart.py`, 277 lines)
- Confirm dropout mask is applied correctly during training and removed during prediction.
- Check that tree weight renormalization after dropout is mathematically correct.

### 2.5 LinearLeafGBDT (`_models/_linear_leaf.py`, 452 lines)
- Review linear model fitting within leaf nodes for numerical stability (singular matrices, regularization).
- Verify prediction path correctly applies linear models vs scalar leaves.

### 2.6 sklearn Wrappers (`_models/_sklearn.py`, 1017 lines)
- Check `get_params()` / `set_params()` roundtrip consistency.
- Verify compatibility with `Pipeline`, `GridSearchCV`, `cross_val_score`.
- Confirm `predict_proba()` output shapes match sklearn conventions.

---

## 3. Data Handling & Preprocessing

### 3.1 Array Binning (`_array.py`, 523 lines)
- Review quantile-based binning for correctness with duplicate values and small datasets.
- Verify missing value encoding (NaN -> bin 255) and categorical feature handling.
- Check `BinnedArray` handles mixed dtypes, integer inputs, and single-column inputs.

### 3.2 Input Validation (`_validation.py`, 473 lines)
- Confirm validation catches: wrong shapes, NaN in targets (when not supported), negative sample weights, invalid hyperparameter combinations.
- Check error messages are informative.

### 3.3 Sampling (`_sampling.py`, 573 lines)
- Review GOSS top-k and random sampling for correctness.
- Verify `MiniBatchIterator` handles non-divisible batch sizes and last-batch edge cases.
- Check memory-mapped array creation and loading for data integrity.

---

## 4. Infrastructure & Reliability

### 4.1 Persistence (`_persistence.py`, 425 lines)
- Review model save/load for all model types: verify no data loss, version compatibility handling, and security (pickle deserialization risks).
- Check backward compatibility strategy for model format changes.

### 4.2 Callbacks (`_callbacks.py`, 405 lines)
- Verify `EarlyStopping` restores best model correctly.
- Check `ModelCheckpoint` writes valid files that can be reloaded.
- Review `LearningRateScheduler` for off-by-one in epoch/iteration counting.

### 4.3 Multi-GPU (`_distributed/`, 1023 lines)
- Review histogram allreduce across GPUs for correctness.
- Check data partitioning logic for uneven splits.
- Verify Ray integration handles worker failures gracefully.

### 4.4 Feature Importance (`_importance.py`, 285 lines)
- Confirm split-based and gain-based importance calculations are correct.
- Verify normalization produces values summing to 1.0.

---

## 5. Public API & Usability

### 5.1 `__init__.py` (462 lines)
- Verify all public exports are intentional and documented.
- Check for accidental exposure of internal modules.
- Confirm `__all__` matches actual exports.

### 5.2 Utilities (`_utils.py`, 1270 lines)
- Review metric implementations (CRPS, Brier score, ECE, calibration curve) against reference formulas.
- Check `suggest_params()` heuristics produce reasonable defaults.
- Verify `cross_val_predict` variants handle stratification and shuffling correctly.

---

## 6. Testing Gaps

### 6.1 Coverage Analysis
- Identify untested code paths in core modules, especially error/edge-case branches.
- Check that all 50+ loss functions have dedicated tests.
- Verify all 8 NaturalBoost distributions have end-to-end tests.

### 6.2 Test Quality
- Review tests for assertion strength (are they testing correct values, not just "no crash"?).
- Check for flaky tests due to random seeds or floating-point tolerances.
- Verify integration tests (`test_integration.py`) cover realistic workflows.

---

## 7. Security & Safety

- Audit `_persistence.py` for unsafe deserialization (arbitrary code execution via pickle).
- Check CUDA kernels for buffer overflows or out-of-bounds memory access.
- Review that no user-supplied strings are passed to `eval()` or `exec()`.

---

## 8. Documentation Accuracy

- Verify code examples in docs (`docs/`) actually run and produce stated outputs.
- Check that API reference (`docs/api/`) matches actual function signatures.
- Review migration guide (`docs/migration/from-xgboost.md`) for accuracy.

---

## 9. CI/CD & Release Readiness

- Review `.github/workflows/unit-tests.yml`: ensure all critical paths are tested in CI.
- Check that GPU tests have a path to run (Modal or self-hosted runners).
- Verify `scripts/pre_launch_check.py` covers all release criteria.
- Confirm `pyproject.toml` metadata (version, classifiers, URLs) is correct for PyPI.

---

## Review Priority Order

| Priority | Area | Rationale |
|----------|------|-----------|
| P0 | Core algorithm correctness (Section 1) | Bugs here produce wrong results silently |
| P0 | CPU/GPU parity (Section 1.2) | Users expect identical results across backends |
| P1 | Model training loops (Section 2.1-2.2) | Directly affects model quality |
| P1 | Loss & distribution math (Sections 1.3-1.4) | Incorrect gradients break training |
| P1 | Security (Section 7) | Pickle deserialization, CUDA memory safety |
| P2 | Data handling (Section 3) | Edge cases in preprocessing |
| P2 | sklearn compatibility (Section 2.6) | Key user-facing integration |
| P2 | Persistence & callbacks (Section 4.1-4.2) | Data loss risks |
| P3 | Testing gaps (Section 6) | Improves confidence in other areas |
| P3 | API surface (Section 5) | Usability and correctness |
| P3 | Documentation (Section 8) | User trust |
| P4 | CI/CD (Section 9) | Process improvement |
