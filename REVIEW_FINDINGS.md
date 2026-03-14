# OpenBoost Code Review Findings

Full code review executed against the review plan. Findings organized by severity.

---

## Critical (P0) -- Bugs That Produce Wrong Results Silently

### CRIT-1: `fit_tree_symmetric` computes total gradient incorrectly
**File:** `src/openboost/_core/_tree.py:1154`
Sums `combined_hist_grad` across ALL features (shape `(n_features, 256)`), giving `n_features * actual_total` instead of the true total. This produces an incorrect parent gain term, causing wrong splits in symmetric trees.

### CRIT-2: NegativeBinomial gradient w.r.t. `mu` uses wrong factor
**File:** `src/openboost/_distributions.py:1303`
Uses `(mu - y) * (1 - p)` where it should be `(mu - y) * p`. Since `p = r/(r+mu)` and `1-p = mu/(r+mu)`, the gradient direction is wrong by a factor of `mu/r`, causing the mean parameter to converge to the wrong value.

### CRIT-3: NegativeBinomial gradient w.r.t. `r` is incomplete
**File:** `src/openboost/_distributions.py:1311`
Missing the `+ mu/(r+mu)` term (equivalently `+ (1 - p)`). The dispersion parameter gradient is incomplete, causing incorrect dispersion estimates.

### CRIT-4: Shared memory OOB for `max_depth > 6` in CUDA histogram kernel
**File:** `src/openboost/_backends/_cuda.py:2927-2948`
`_build_histogram_shared_kernel` allocates shared memory for 16 nodes per pass and does at most 2 passes (32 nodes). At `max_depth=7`, depth 6 has 64 nodes, causing out-of-bounds shared memory access -- undefined behavior / silent corruption.

### CRIT-5: Symmetric GPU tree reuses one global histogram for all depths
**File:** `src/openboost/_backends/_cuda.py:3388,3405`
`build_tree_symmetric_gpu_native` builds ONE histogram and reuses it for ALL depth levels. After the first split, deeper splits should use conditional distributions per leaf, not the global distribution. This produces incorrect trees.

### CRIT-6: Batch CPU training gradient update is a no-op
**File:** `src/openboost/_core/_tree.py:889-890`
The expression `2 * (pred - (pred - grad / 2))` simplifies to just `grad`. Every boosting round uses the same initial gradient, making batch CPU boosting non-functional.

### CRIT-7: Train vs transform binning mismatch
**File:** `src/openboost/_array.py:381,386 vs 146`
Training uses `np.digitize` (no clip), while `transform()` uses `np.searchsorted` clipped to `len(edges) - 1`. Maximum bin indices differ at upper boundary, causing train/test prediction inconsistency.

### CRIT-8: `CustomDistribution` numerical gradients are w.r.t. constrained params, not raw params
**File:** `src/openboost/_distributions.py:1525-1561`
`_numerical_gradient` perturbs constrained parameters but the interface requires gradients w.r.t. raw parameters. The chain rule through the link function is not applied, producing incorrect gradients for any non-identity link.

---

## High (P1) -- Significant Bugs or Security Issues

### HIGH-1: float32 (GPU) vs float64 (CPU) in split-finding
**Files:** `_backends/_cpu.py:138,296-301` vs `_backends/_cuda.py:1010,1465`
CPU computes gains in float64, GPU in float32. For close gains, different splits are selected. Users expect identical results across backends.

### HIGH-2: CUDA per-feature total grad/hess divergence in level splits
**File:** `src/openboost/_backends/_cuda.py:2647-2656`
Each feature's thread computes its own `total_grad`/`total_hess` from its histogram. Float32 atomic accumulation introduces per-feature differences, causing gain comparison across features to use different parent baselines. Can select suboptimal features.

### HIGH-3: DART tree weight corruption via cumulative renormalization
**File:** `src/openboost/_models/_dart.py:160-168`
The `k/(k+1)` rescaling permanently mutates `tree_weights_` in-place every round a tree is dropped. Weights compound across rounds, driving early trees toward zero. The DART paper normalizes at prediction time, not by mutating stored weights.

### HIGH-4: LinearLeafGBDT leaf identification via float equality
**File:** `src/openboost/_models/_linear_leaf.py:90-98`
Training uses `np.isclose` (line 294) to assign samples to leaves, but prediction uses exact `==` (line 94). Float precision differences (especially CPU vs GPU) cause lookup failures, silently defaulting to `leaf_idx = 0` for mismatched samples.

### HIGH-5: `ModelCheckpoint` uses raw `pickle.dump`, bypassing persistence system
**File:** `src/openboost/_callbacks.py:296-299`
Models with CUDA arrays will fail to pickle or produce files unloadable on CPU-only machines. Should use `model.save()` instead.

### HIGH-6: `_loss_fn` and `distribution_` not restored after model load
**File:** `src/openboost/_persistence.py:259-263,345`
These are skipped during serialization with comments saying they'll be recreated, but `_from_state_dict` has no code to recreate them. Only works if the subclass defines `_post_load`, which is not guaranteed.

### HIGH-7: GPU loss kernels can be `None` with no lazy-init fallback
**File:** `src/openboost/_loss.py:183-184,271-272,359,443,542,620,700,798`
`_ensure_*_kernel()` functions exist but are never called from GPU dispatch paths. If eager CUDA compilation fails, calling the GPU function raises `TypeError: 'NoneType' is not subscriptable`.

### HIGH-8: Security -- pickle deserialization with no runtime warning
**File:** `src/openboost/_persistence.py:388`
`joblib.load(path)` executes arbitrary code. Only a docstring warning exists -- no runtime `UserWarning` is emitted. `_from_state_dict` also allows `setattr` from untrusted data (line 281-314).

### HIGH-9: `LeafWiseGrowth` ignores missing values and categorical features
**File:** `src/openboost/_core/_growth.py:680-686`
Does not pass `has_missing`, `is_categorical`, or `n_categories` to `find_node_splits`. Missing values and categoricals are silently ignored in leaf-wise growth.

### HIGH-10: `SymmetricGrowth` ignores missing values
**File:** `src/openboost/_core/_growth.py:851,871`
Does not pass `has_missing` to `find_node_splits`. Missing samples (bin 255) always go right with no learned direction.

### HIGH-11: Ray distributed training uses per-shard bin edges
**File:** `src/openboost/_distributed/_ray.py:99-107`
Each Ray worker independently computes bin edges from its local shard. Bin `k` on worker 0 may correspond to a different value range than bin `k` on worker 1, making histogram allreduce semantically incorrect.

### HIGH-12: `fit_tree_multigpu` only builds depth-1 trees
**File:** `src/openboost/_distributed/_multigpu.py:584-669`
`_build_tree_from_global_histogram` always produces 3-node trees regardless of `max_depth`. Multi-GPU tree building is effectively non-functional for real training.

---

## Medium (P2) -- Correctness Issues With Limited Scope

### MED-1: `SymmetricGrowth` does not apply L1 regularization to leaf values
**File:** `src/openboost/_core/_growth.py:890`
Uses `-sum_grad / (sum_hess + reg_lambda)` without L1 soft-thresholding (`reg_alpha`). Both other growth strategies apply L1 correctly.

### MED-2: `colsample_bytree` stored in config but never used by any growth strategy
**File:** `src/openboost/_core/_growth.py:163`
Declared in `GrowthConfig` and set in `fit_tree`, but no growth strategy references it. Column subsampling is a no-op.

### MED-3: Subsample implemented by zeroing gradients, not excluding samples
**File:** `src/openboost/_core/_tree.py:417-426`
Non-sampled gradients/hessians are set to zero but samples still participate in histogram building. No performance benefit from subsampling, and `min_child_weight` checks count zero-weight samples.

### MED-4: `LevelWiseGrowth` reports tree depth one too high when last level has no valid splits
**File:** `src/openboost/_core/_growth.py:517`
`actual_depth = depth + 1` is set before checking if any splits exist at that depth.

### MED-5: Missing value direction override in partitioning
**File:** `src/openboost/_core/_primitives.py:576-579`
The tree-wide `missing_go_left` array (initialized to all-True) overrides the per-split learned direction for nodes not yet split.

### MED-6: StudentT `df` gradient is a hard-coded constant
**File:** `src/openboost/_distributions.py:935-936`
`grad_df = 0.01 * np.ones_like(y)` -- not derived from the NLL. The model cannot learn tail heaviness from data.

### MED-7: `get_loss_function` does not pass `huber_delta` through
**File:** `src/openboost/_loss.py:65`
Unlike quantile and tweedie which get lambda wrappers, huber always uses `delta=1.0`.

### MED-8: `validate_eval_set` accesses `.shape[1]` on `BinnedArray`
**File:** `src/openboost/_validation.py:327`
`BinnedArray` is a dataclass without a `.shape` attribute. Should use `.n_features`.

### MED-9: `MultiClassGradientBoosting` has no input validation
**File:** `src/openboost/_models/_boosting.py:1017-1034`
`fit()` does not call `validate_X`, `validate_y`, or `validate_hyperparameters`. `predict_proba()` does not call `validate_predict_input`.

### MED-10: `EarlyStopping` only snapshots `trees_` and `tree_weights_`
**File:** `src/openboost/_callbacks.py:183`
Other model state (`base_score_`, `feature_importances_`, internal counters) is not saved, so the "restored" model may have inconsistent state.

### MED-11: `CallbackManager.on_round_end` short-circuits on first `False`
**File:** `src/openboost/_callbacks.py:397-399`
If `EarlyStopping` is listed before `HistoryCallback`, the history callback won't record the final round's metrics.

### MED-12: `cross_val_predict_proba` uses `KFold` instead of `StratifiedKFold`
**File:** `src/openboost/_utils.py:1101`
For imbalanced classification, a fold might contain no positive samples.

### MED-13: `feature_importances_` silently falls back from gain/cover to frequency
**File:** `src/openboost/_importance.py:165-178`
When trees lack `split_gains` or `node_counts`, falls back to frequency counting with no warning.

### MED-14: `LinearLeafTree` causes `AttributeError` in feature importance
**File:** `src/openboost/_importance.py:99-121`
`_get_trees_flat` does not unwrap `LinearLeafTree.tree_structure`, so `_accumulate_importance` tries to access `tree.n_nodes` on the wrapper object.

### MED-15: GPU sample weights silently ignored
**File:** `src/openboost/_models/_boosting.py:627-631`
No warning or error raised when `sample_weight` is passed with GPU backend active.

### MED-16: `DART` `sample_type='weighted'` falls back silently to uniform
**File:** `src/openboost/_models/_dart.py:192-194`
User gets uniform dropout with no warning.

### MED-17: sklearn `OpenBoostClassifier` does not pass `batch_size` for multi-class
**File:** `src/openboost/_models/_sklearn.py:512-528`
`batch_size` is silently ignored for multi-class classification.

### MED-18: No serialization version number
**File:** `src/openboost/_persistence.py:215-271`
State dict has no version field. Format changes will silently produce incorrect results on old models.

### MED-19: Softplus inverse link overflows for large inputs
**File:** `src/openboost/_distributions.py:1440`
`log(exp(x) - 1)` overflows for `x > ~700`. Should use `np.where(x > 20, x, np.log(np.exp(x) - 1))`.

### MED-20: Tweedie deviance clips `y=0` to `1e-10` for zero-inflated data
**File:** `src/openboost/_distributions.py:1122-1123`
Introduces systematic bias in the deviance calculation for the exact use case (zero-inflated data) the distribution is designed for.

### MED-21: `LinearLeafGBDT` stores reference to full training data
**File:** `src/openboost/_models/_linear_leaf.py:205`
`self._X_raw = X` persists after fitting, causing the entire training dataset to be serialized with the model.

---

## Low (P3) -- Minor Issues, Performance, Code Quality

### LOW-1: CUDA predict kernels recompiled on every call
**File:** `src/openboost/_core/_predict.py:69-77,83-92`
`_fill_cuda` and `_add_inplace_cuda` define `@cuda.jit` kernels inside the function body.

### LOW-2: Non-deterministic partition order from CUDA atomic scatter
**File:** `src/openboost/_backends/_cuda.py:643-667`
`left_out` and `right_out` contain correct elements but in non-deterministic order.

### LOW-3: CUDA `reduce_sum_cuda` returns uninitialized result for empty input
**File:** `src/openboost/_backends/_cuda.py:330-355`

### LOW-4: No guard for zero features in histogram/split code
**Files:** `_backends/_cpu.py`, `_backends/_cuda.py`
`np.argmax` on empty array raises `ValueError`.

### LOW-5: `_compute_leaf_sums_kernel` uses float32 atomics (non-deterministic)
**File:** `src/openboost/_backends/_cuda.py:2806-2807`
Leaf values can differ across runs at float32 precision.

### LOW-6: Module-level `_BACKEND` state not thread-safe
**File:** `src/openboost/_backends/__init__.py:12`

### LOW-7: Nested `@cuda.jit` definitions inside function body
**File:** `src/openboost/_backends/_cuda.py:3373,3449`
Can cause recompilation issues with different argument types.

### LOW-8: `suggest_params` ignores the `y` argument entirely
**File:** `src/openboost/_utils.py:917-969`

### LOW-9: `suggest_params` returns `n_estimators` but models use `n_trees`
**File:** `src/openboost/_utils.py:925-931`

### LOW-10: CRPS docstring formula is wrong (implementation is correct)
**File:** `src/openboost/_utils.py:409`

### LOW-11: `feature_importances_` property recomputes on every access
**File:** `src/openboost/_models/_sklearn.py:308-312,594-598`
sklearn convention expects stable fitted attributes.

### LOW-12: All models initialize predictions to zero (no base score)
**Files:** `_dart.py:119`, `_boosting.py:747`, `_gam.py:155`, `_linear_leaf.py:228`
For classification, `log(p/(1-p))` would be a better initial prediction.

### LOW-13: `subtract_histogram` imported twice under different names
**File:** `src/openboost/_core/__init__.py:49`

### LOW-14: `fit_tree_gpu_native` returns `Tree` while `fit_tree` returns `TreeStructure`
**File:** `src/openboost/_core/_tree.py:200,273`

### LOW-15: Categorical feature transform is O(n) Python loop
**File:** `src/openboost/_array.py:117-133`

### LOW-16: `CustomDistribution` JAX gradients computed sample-by-sample
**File:** `src/openboost/_distributions.py:1582-1601`
No `jax.vmap`, kernels re-traced on every call.

### LOW-17: No CPU equivalent for categorical prediction
**File:** `_backends/_cpu.py` -- `predict_with_categorical_cpu` is absent.

---

## Testing Gaps

| Area | Status |
|------|--------|
| `huber_gradient` | No dedicated test anywhere |
| `NaturalBoostLogNormal` | No end-to-end test |
| `NaturalBoostGamma` | No end-to-end test |
| `NaturalBoostPoisson` | No end-to-end test |
| `OpenBoostGAM` | No functional CPU test (only persistence/GPU tests) |
| Sampling module (`goss_sample`, `MiniBatchIterator`, etc.) | No `test_sampling.py` |
| GPU tests in CI | Never run (CPU-only CI) |
| Loss integration tests | Only check "no NaN", not correctness |
| `DistributionalGBDT` / `DART` | No early stopping support or tests |
| `MultiClassGradientBoosting` | No input validation tests |

## CI/CD Issues

| Issue | Details |
|-------|---------|
| No GPU CI | All CUDA code untested in CI |
| No Windows CI | Only Ubuntu + macOS |
| `scipy` not declared as dependency | Required by distributions at runtime |
| Publish workflow has no test gate | Broken releases can be pushed to PyPI |
| PyPI classifier says "Beta" for 1.0.0rc1 | Minor metadata inconsistency |

## Documentation Issues

| Issue | Location |
|-------|----------|
| `compute_feature_importances` docs pass `model.trees_` instead of `model` | `docs/quickstart.md:214-222` |
| `sample()` shape comment is wrong | `docs/quickstart.md:152` |
| Persistence docs show `pickle.load` with no security caveat | `docs/user-guide/model-persistence.md:56` |

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| Critical (P0) | 8 |
| High (P1) | 12 |
| Medium (P2) | 21 |
| Low (P3) | 17 |
| **Total findings** | **58** |
