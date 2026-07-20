# OpenBoost Code Review Findings

Full code review executed against the review plan. Findings organized by severity.

> **Post-audit refresh (2026-07-20):** every finding below is annotated with its
> current status — **FIXED** (with the commit or work area that fixed it) or
> **OPEN**. CRIT-1/2/4/7 were fixed in the earlier remediation commits
> (b9c7efc follow-up work / 30013b1). CRIT-3, CRIT-5 (by gating), CRIT-6,
> CRIT-8 (including the hessian second-order term) and the JAX chain rule were
> fixed in the 2026-07-20 remediation pass. Original findings text is preserved
> unchanged.

---

## Critical (P0) -- Bugs That Produce Wrong Results Silently

### CRIT-1: `fit_tree_symmetric` computes total gradient incorrectly
**File:** `src/openboost/_core/_tree.py:1154`
Sums `combined_hist_grad` across ALL features (shape `(n_features, 256)`), giving `n_features * actual_total` instead of the true total. This produces an incorrect parent gain term, causing wrong splits in symmetric trees.

**Status (2026-07-20): FIXED** (earlier commits b9c7efc/30013b1). `fit_tree_symmetric` now delegates to `SymmetricGrowth`, which aggregates per-leaf gains from per-leaf histograms; the erroneous all-features summation no longer exists.

### CRIT-2: NegativeBinomial gradient w.r.t. `mu` uses wrong factor
**File:** `src/openboost/_distributions.py:1303`
Uses `(mu - y) * (1 - p)` where it should be `(mu - y) * p`. Since `p = r/(r+mu)` and `1-p = mu/(r+mu)`, the gradient direction is wrong by a factor of `mu/r`, causing the mean parameter to converge to the wrong value.

**Status (2026-07-20): FIXED** (earlier commits b9c7efc/30013b1). Code now computes `grad_mu_raw = (mu - y) * p`; the stale derivation comment was corrected in the 2026-07-20 distributions pass, and the gradient is verified against finite differences in `tests/test_distribution_gradients.py`.

### CRIT-3: NegativeBinomial gradient w.r.t. `r` is incomplete
**File:** `src/openboost/_distributions.py:1311`
Missing the `+ mu/(r+mu)` term (equivalently `+ (1 - p)`). The dispersion parameter gradient is incomplete, causing incorrect dispersion estimates.

**Status (2026-07-20): FIXED** (2026-07-20 distributions pass). The r-gradient was rederived as `grad_r_raw = -r*(digamma(y+r)-digamma(r)+log(p)+(1-p)-y/(r+mu))` (the old code also had a wrong overall sign) and is verified against central finite differences and a closed-form check in `tests/test_distribution_gradients.py`.

### CRIT-4: Shared memory OOB for `max_depth > 6` in CUDA histogram kernel
**File:** `src/openboost/_backends/_cuda.py:2927-2948`
`_build_histogram_shared_kernel` allocates shared memory for 16 nodes per pass and does at most 2 passes (32 nodes). At `max_depth=7`, depth 6 has 64 nodes, causing out-of-bounds shared memory access -- undefined behavior / silent corruption.

**Status (2026-07-20): FIXED** (earlier commits b9c7efc/30013b1, plus later GPU work). The shared kernel now runs `ceil(n_nodes / 16)` passes with an in-kernel `n_nodes_in_pass` bound check, and deep levels can route through a single-pass global-atomic histogram path.

### CRIT-5: Symmetric GPU tree reuses one global histogram for all depths
**File:** `src/openboost/_backends/_cuda.py:3388,3405`
`build_tree_symmetric_gpu_native` builds ONE histogram and reuses it for ALL depth levels. After the first split, deeper splits should use conditional distributions per leaf, not the global distribution. This produces incorrect trees.

**Status (2026-07-20): FIXED BY GATING** (2026-07-20 pass). The defective GPU kernel was **not rewritten**: the kernel docstring now documents the defect explicitly, and `fit_tree_symmetric_gpu_native` falls back to the correct CPU `SymmetricGrowth` builder unless `OPENBOOST_EXPERIMENTAL_SYMMETRIC_GPU=1` is set. A leaf-aware GPU kernel remains future work.

### CRIT-6: Batch CPU training gradient update is a no-op
**File:** `src/openboost/_core/_tree.py:889-890`
The expression `2 * (pred - (pred - grad / 2))` simplifies to just `grad`. Every boosting round uses the same initial gradient, making batch CPU boosting non-functional.

**Status (2026-07-20): FIXED** (2026-07-20 pass). `fit_trees_batch` now recomputes per-round gradients via the resolved loss function and `y` (and raises a clear error when non-MSE losses are used without passing `y`).

### CRIT-7: Train vs transform binning mismatch
**File:** `src/openboost/_array.py:381,386 vs 146`
Training uses `np.digitize` (no clip), while `transform()` uses `np.searchsorted` clipped to `len(edges) - 1`. Maximum bin indices differ at upper boundary, causing train/test prediction inconsistency.

**Status (2026-07-20): FIXED** (earlier commits b9c7efc/30013b1). Both fit-time binning and `transform()` now use the same `np.searchsorted(edges, ..., side='right')` logic; parity is covered by `tests/test_binning_correctness.py`.

### CRIT-8: `CustomDistribution` numerical gradients are w.r.t. constrained params, not raw params
**File:** `src/openboost/_distributions.py:1525-1561`
`_numerical_gradient` perturbs constrained parameters but the interface requires gradients w.r.t. raw parameters. The chain rule through the link function is not applied, producing incorrect gradients for any non-identity link.

**Status (2026-07-20): FIXED** (2026-07-20 distributions pass). The numerical path now applies the full chain rule — `grad_raw = grad_c * link'` and the second-order hessian term `hess_raw = hess_c * (link')^2 + grad_c * link''` via a new `_link_second_derivative` helper. The JAX path was also fixed to differentiate `nll(y, link(raw))` w.r.t. raw parameters (exact raw-space hessian). Verified in `tests/test_distribution_gradients.py`, including a JAX-vs-numerical agreement test.

---

## High (P1) -- Significant Bugs or Security Issues

### HIGH-1: float32 (GPU) vs float64 (CPU) in split-finding
**Files:** `_backends/_cpu.py:138,296-301` vs `_backends/_cuda.py:1010,1465`
CPU computes gains in float64, GPU in float32. For close gains, different splits are selected. Users expect identical results across backends.

**Status (2026-07-20): OPEN** (accepted precision tradeoff). GPU split-gain arithmetic now uses float64 accumulators in the split kernels, but GPU histograms remain float32 (deliberately, for atomic-add throughput), so bitwise CPU/GPU split parity is still not guaranteed.

### HIGH-2: CUDA per-feature total grad/hess divergence in level splits
**File:** `src/openboost/_backends/_cuda.py:2647-2656`
Each feature's thread computes its own `total_grad`/`total_hess` from its histogram. Float32 atomic accumulation introduces per-feature differences, causing gain comparison across features to use different parent baselines. Can select suboptimal features.

**Status (2026-07-20): FIXED.** `_find_level_splits_kernel` now computes the node totals once (thread 0, float64, from feature 0's histogram), broadcasts them via shared memory, and every feature uses the same parent baseline.

### HIGH-3: DART tree weight corruption via cumulative renormalization
**File:** `src/openboost/_models/_dart.py:160-168`
The `k/(k+1)` rescaling permanently mutates `tree_weights_` in-place every round a tree is dropped. Weights compound across rounds, driving early trees toward zero. The DART paper normalizes at prediction time, not by mutating stored weights.

**Status (2026-07-20): FIXED.** Trees are stored with weight 1.0 and the `k/(k+1)` normalization is applied at prediction/aggregation time; stored weights are no longer mutated.

### HIGH-4: LinearLeafGBDT leaf identification via float equality
**File:** `src/openboost/_models/_linear_leaf.py:90-98`
Training uses `np.isclose` (line 294) to assign samples to leaves, but prediction uses exact `==` (line 94). Float precision differences (especially CPU vs GPU) cause lookup failures, silently defaulting to `leaf_idx = 0` for mismatched samples.

**Status (2026-07-20): FIXED.** Both training and prediction now route samples structurally through the tree (`_route_to_leaves` on node indices); no float-value matching remains.

### HIGH-5: `ModelCheckpoint` uses raw `pickle.dump`, bypassing persistence system
**File:** `src/openboost/_callbacks.py:296-299`
Models with CUDA arrays will fail to pickle or produce files unloadable on CPU-only machines. Should use `model.save()` instead.

**Status (2026-07-20): FIXED.** `ModelCheckpoint` now calls `model.save()` and only falls back to pickle (with an explicit `UserWarning`) when the model has no `save()`.

### HIGH-6: `_loss_fn` and `distribution_` not restored after model load
**File:** `src/openboost/_persistence.py:259-263,345`
These are skipped during serialization with comments saying they'll be recreated, but `_from_state_dict` has no code to recreate them. Only works if the subclass defines `_post_load`, which is not guaranteed.

**Status (2026-07-20): FIXED.** `_from_state_dict` now reconstructs `_loss_fn` (via `get_loss_function`) and `distribution_` (via `get_distribution`) directly, and additionally invokes `_post_load` when defined.

### HIGH-7: GPU loss kernels can be `None` with no lazy-init fallback
**File:** `src/openboost/_loss.py:183-184,271-272,359,443,542,620,700,798`
`_ensure_*_kernel()` functions exist but are never called from GPU dispatch paths. If eager CUDA compilation fails, calling the GPU function raises `TypeError: 'NoneType' is not subscriptable`.

**Status (2026-07-20): FIXED.** Every GPU loss dispatch path now calls its `_ensure_*_kernel()` lazy initializer before kernel launch.

### HIGH-8: Security -- pickle deserialization with no runtime warning
**File:** `src/openboost/_persistence.py:388`
`joblib.load(path)` executes arbitrary code. Only a docstring warning exists -- no runtime `UserWarning` is emitted. `_from_state_dict` also allows `setattr` from untrusted data (line 281-314).

**Status (2026-07-20): FIXED.** `load()` now emits a runtime `UserWarning` about untrusted files, and the state dict carries a `_serialization_version` that is validated on load (see MED-18).

### HIGH-9: `LeafWiseGrowth` ignores missing values and categorical features
**File:** `src/openboost/_core/_growth.py:680-686`
Does not pass `has_missing`, `is_categorical`, or `n_categories` to `find_node_splits`. Missing values and categoricals are silently ignored in leaf-wise growth.

**Status (2026-07-20): FIXED.** `LeafWiseGrowth` now threads `has_missing`, `is_categorical`, and `n_categories` through both root and child split searches.

### HIGH-10: `SymmetricGrowth` ignores missing values
**File:** `src/openboost/_core/_growth.py:851,871`
Does not pass `has_missing` to `find_node_splits`. Missing samples (bin 255) always go right with no learned direction.

**Status (2026-07-20): OPEN** (partially mitigated). `SymmetricGrowth.grow` accepts `has_missing`, but the level-split search still does not learn a missing-value direction: bin 255 deterministically routes right. The behavior is now documented and consistent between training, partitioning, and prediction, but remains unlearned.

### HIGH-11: Ray distributed training uses per-shard bin edges
**File:** `src/openboost/_distributed/_ray.py:99-107`
Each Ray worker independently computes bin edges from its local shard. Bin `k` on worker 0 may correspond to a different value range than bin `k` on worker 1, making histogram allreduce semantically incorrect.

**Status (2026-07-20): FIXED.** The driver computes global bin edges once and passes them to every `RayWorker`, so all shards bin identically.

### HIGH-12: `fit_tree_multigpu` only builds depth-1 trees
**File:** `src/openboost/_distributed/_multigpu.py:584-669`
`_build_tree_from_global_histogram` always produces 3-node trees regardless of `max_depth`. Multi-GPU tree building is effectively non-functional for real training.

**Status (2026-07-20): FIXED.** `_build_tree_from_global_histogram` now implements full level-wise building up to `max_depth` from the aggregated histograms.

---

## Medium (P2) -- Correctness Issues With Limited Scope

### MED-1: `SymmetricGrowth` does not apply L1 regularization to leaf values
**File:** `src/openboost/_core/_growth.py:890`
Uses `-sum_grad / (sum_hess + reg_lambda)` without L1 soft-thresholding (`reg_alpha`). Both other growth strategies apply L1 correctly.

**Status (2026-07-20): FIXED.** SymmetricGrowth leaf values now apply the same `reg_alpha` soft-thresholding as the other strategies.

### MED-2: `colsample_bytree` stored in config but never used by any growth strategy
**File:** `src/openboost/_core/_growth.py:163`
Declared in `GrowthConfig` and set in `fit_tree`, but no growth strategy references it. Column subsampling is a no-op.

**Status (2026-07-20): FIXED.** Growth now selects a per-tree feature subset when `colsample_bytree < 1.0`.

### MED-3: Subsample implemented by zeroing gradients, not excluding samples
**File:** `src/openboost/_core/_tree.py:417-426`
Non-sampled gradients/hessians are set to zero but samples still participate in histogram building. No performance benefit from subsampling, and `min_child_weight` checks count zero-weight samples.

**Status (2026-07-20): OPEN** (acknowledged design choice). The zeroing approach is now documented in-code as mathematically correct (zero-weight samples contribute nothing to grad/hess sums, including `min_child_weight`), but the performance benefit of true row exclusion is still not realized; a TODO records the intended improvement.

### MED-4: `LevelWiseGrowth` reports tree depth one too high when last level has no valid splits
**File:** `src/openboost/_core/_growth.py:517`
`actual_depth = depth + 1` is set before checking if any splits exist at that depth.

**Status (2026-07-20): FIXED.** `actual_depth` is only advanced when at least one valid split was found at that depth.

### MED-5: Missing value direction override in partitioning
**File:** `src/openboost/_core/_primitives.py:576-579`
The tree-wide `missing_go_left` array (initialized to all-True) overrides the per-split learned direction for nodes not yet split.

**Status (2026-07-20): FIXED.** Partitioning now uses the per-split learned `missing_go_left` as the primary source, falling back to the tree-wide array only when the split lacks the attribute.

### MED-6: StudentT `df` gradient is a hard-coded constant
**File:** `src/openboost/_distributions.py:935-936`
`grad_df = 0.01 * np.ones_like(y)` -- not derived from the NLL. The model cannot learn tail heaviness from data.

**Status (2026-07-20): FIXED** (2026-07-20 distributions pass). The df gradient is now the exact `d(NLL)/dnu` (digamma terms) times the true softplus+2 link derivative, verified against finite differences in `tests/test_distribution_gradients.py`.

### MED-7: `get_loss_function` does not pass `huber_delta` through
**File:** `src/openboost/_loss.py:65`
Unlike quantile and tweedie which get lambda wrappers, huber always uses `delta=1.0`.

**Status (2026-07-20): FIXED.** `huber_delta` (or `delta`) is now forwarded via a lambda wrapper like quantile/tweedie.

### MED-8: `validate_eval_set` accesses `.shape[1]` on `BinnedArray`
**File:** `src/openboost/_validation.py:327`
`BinnedArray` is a dataclass without a `.shape` attribute. Should use `.n_features`.

**Status (2026-07-20): FIXED.** Uses `.n_features` when present, `.shape[1]` otherwise.

### MED-9: `MultiClassGradientBoosting` has no input validation
**File:** `src/openboost/_models/_boosting.py:1017-1034`
`fit()` does not call `validate_X`, `validate_y`, or `validate_hyperparameters`. `predict_proba()` does not call `validate_predict_input`.

**Status (2026-07-20): FIXED.** `fit()` now calls `validate_X`/`validate_y`/`validate_eval_set` and prediction paths call `validate_predict_input`. (Dedicated tests for this are still missing — see Testing Gaps.)

### MED-10: `EarlyStopping` only snapshots `trees_` and `tree_weights_`
**File:** `src/openboost/_callbacks.py:183`
Other model state (`base_score_`, `feature_importances_`, internal counters) is not saved, so the "restored" model may have inconsistent state.

**Status (2026-07-20): FIXED.** The snapshot now includes trees, DART weights, tree weights, and `base_score_`.

### MED-11: `CallbackManager.on_round_end` short-circuits on first `False`
**File:** `src/openboost/_callbacks.py:397-399`
If `EarlyStopping` is listed before `HistoryCallback`, the history callback won't record the final round's metrics.

**Status (2026-07-20): FIXED.** All callbacks are invoked every round; the stop decision is aggregated afterwards.

### MED-12: `cross_val_predict_proba` uses `KFold` instead of `StratifiedKFold`
**File:** `src/openboost/_utils.py:1101`
For imbalanced classification, a fold might contain no positive samples.

**Status (2026-07-20): FIXED.** `cross_val_predict_proba` now uses `StratifiedKFold`.

### MED-13: `feature_importances_` silently falls back from gain/cover to frequency
**File:** `src/openboost/_importance.py:165-178`
When trees lack `split_gains` or `node_counts`, falls back to frequency counting with no warning.

**Status (2026-07-20): FIXED.** A `UserWarning` is emitted on fallback.

### MED-14: `LinearLeafTree` causes `AttributeError` in feature importance
**File:** `src/openboost/_importance.py:99-121`
`_get_trees_flat` does not unwrap `LinearLeafTree.tree_structure`, so `_accumulate_importance` tries to access `tree.n_nodes` on the wrapper object.

**Status (2026-07-20): FIXED.** `_get_trees_flat` unwraps `tree_structure` wrappers.

### MED-15: GPU sample weights silently ignored
**File:** `src/openboost/_models/_boosting.py:627-631`
No warning or error raised when `sample_weight` is passed with GPU backend active.

**Status (2026-07-20): FIXED.** A `UserWarning` ("sample_weight is not supported on GPU backend and will be ignored") is now raised; documented as a known 1.0.0rc1 limitation.

### MED-16: `DART` `sample_type='weighted'` falls back silently to uniform
**File:** `src/openboost/_models/_dart.py:192-194`
User gets uniform dropout with no warning.

**Status (2026-07-20): FIXED.** The fallback now emits a `UserWarning`; unknown values raise `ValueError`.

### MED-17: sklearn `OpenBoostClassifier` does not pass `batch_size` for multi-class
**File:** `src/openboost/_models/_sklearn.py:512-528`
`batch_size` is silently ignored for multi-class classification.

**Status (2026-07-20): FIXED.** `batch_size` is forwarded on both binary and multi-class paths.

### MED-18: No serialization version number
**File:** `src/openboost/_persistence.py:215-271`
State dict has no version field. Format changes will silently produce incorrect results on old models.

**Status (2026-07-20): FIXED.** State dicts now carry `_serialization_version` (currently 1); loading warns on missing versions and errors on newer-than-supported versions.

### MED-19: Softplus inverse link overflows for large inputs
**File:** `src/openboost/_distributions.py:1440`
`log(exp(x) - 1)` overflows for `x > ~700`. Should use `np.where(x > 20, x, np.log(np.exp(x) - 1))`.

**Status (2026-07-20): FIXED.** The inverse softplus uses the suggested stable branch (`np.where(x > 20, x, log(expm1(min(x, 20))))`).

### MED-20: Tweedie deviance clips `y=0` to `1e-10` for zero-inflated data
**File:** `src/openboost/_distributions.py:1122-1123`
Introduces systematic bias in the deviance calculation for the exact use case (zero-inflated data) the distribution is designed for.

**Status (2026-07-20): FIXED.** `_compute_deviance` handles `y=0` exactly via `np.where` (deviance `2*mu^(2-p)/(2-p)`). Additionally, the 2026-07-20 pass made `Tweedie.nll()` a proper log-likelihood (exact `-log P(Y=0)` mass at zero, Dunn & Smyth series density for `y>0`) and gave `Tweedie.quantile()` support-respecting behavior at the zero mass; training gradients intentionally still descend the deviance-based objective.

### MED-21: `LinearLeafGBDT` stores reference to full training data
**File:** `src/openboost/_models/_linear_leaf.py:205`
`self._X_raw = X` persists after fitting, causing the entire training dataset to be serialized with the model.

**Status (2026-07-20): FIXED.** `_X_raw` is released (`None`) at the end of `fit()`.

---

## Low (P3) -- Minor Issues, Performance, Code Quality

### LOW-1: CUDA predict kernels recompiled on every call
**File:** `src/openboost/_core/_predict.py:69-77,83-92`
`_fill_cuda` and `_add_inplace_cuda` define `@cuda.jit` kernels inside the function body.

**Status (2026-07-20): FIXED.** Kernels are cached at module level behind lazy getters.

### LOW-2: Non-deterministic partition order from CUDA atomic scatter
**File:** `src/openboost/_backends/_cuda.py:643-667`
`left_out` and `right_out` contain correct elements but in non-deterministic order.

**Status (2026-07-20): OPEN** (documented). The non-determinism is now called out in the kernel docs; element sets remain correct.

### LOW-3: CUDA `reduce_sum_cuda` returns uninitialized result for empty input
**File:** `src/openboost/_backends/_cuda.py:330-355`

**Status (2026-07-20): FIXED.** Empty input returns an explicit 0.0.

### LOW-4: No guard for zero features in histogram/split code
**Files:** `_backends/_cpu.py`, `_backends/_cuda.py`
`np.argmax` on empty array raises `ValueError`.

**Status (2026-07-20): FIXED.** The CPU split path guards `n_features == 0` (returns no-split) before any argmax.

### LOW-5: `_compute_leaf_sums_kernel` uses float32 atomics (non-deterministic)
**File:** `src/openboost/_backends/_cuda.py:2806-2807`
Leaf values can differ across runs at float32 precision.

**Status (2026-07-20): FIXED.** Leaf sum outputs are float64 atomics; deep-level leaf sums can also be derived from histograms instead of per-sample atomics.

### LOW-6: Module-level `_BACKEND` state not thread-safe
**File:** `src/openboost/_backends/__init__.py:12`

**Status (2026-07-20): FIXED.** Backend get/set/auto-detect are guarded by `_BACKEND_LOCK`; `backend_context` is documented as a process-global switch.

### LOW-7: Nested `@cuda.jit` definitions inside function body
**File:** `src/openboost/_backends/_cuda.py:3373,3449`
Can cause recompilation issues with different argument types.

**Status (2026-07-20): FIXED.** The affected kernels are now defined at module level.

### LOW-8: `suggest_params` ignores the `y` argument entirely
**File:** `src/openboost/_utils.py:917-969`

**Status (2026-07-20): FIXED.** `y` now informs class-count and imbalance detection.

### LOW-9: `suggest_params` returns `n_estimators` but models use `n_trees`
**File:** `src/openboost/_utils.py:925-931`

**Status (2026-07-20): FIXED.** A `style` parameter selects naming: `'sklearn'` (default, `n_estimators`) or `'core'` (`n_trees`).

### LOW-10: CRPS docstring formula is wrong (implementation is correct)
**File:** `src/openboost/_utils.py:409`

**Status (2026-07-20): FIXED.** The docstring now states the correct Gaussian CRPS formula.

### LOW-11: `feature_importances_` property recomputes on every access
**File:** `src/openboost/_models/_sklearn.py:308-312,594-598`
sklearn convention expects stable fitted attributes.

**Status (2026-07-20): FIXED.** Importances are computed once at fit time and cached.

### LOW-12: All models initialize predictions to zero (no base score)
**Files:** `_dart.py:119`, `_boosting.py:747`, `_gam.py:155`, `_linear_leaf.py:228`
For classification, `log(p/(1-p))` would be a better initial prediction.

**Status (2026-07-20): FIXED.** `GradientBoosting`, `DART`, and `OpenBoostGAM` initialize `base_score_` (log-odds for logloss, mean for regression) and apply it in predict.

### LOW-13: `subtract_histogram` imported twice under different names
**File:** `src/openboost/_core/__init__.py:49`

**Status (2026-07-20): FIXED.** Single import/export remains.

### LOW-14: `fit_tree_gpu_native` returns `Tree` while `fit_tree` returns `TreeStructure`
**File:** `src/openboost/_core/_tree.py:200,273`

**Status (2026-07-20): OPEN.** The return-type asymmetry persists (`Tree` keeps arrays GPU-resident by design; conversion is lazy via `to_arrays()`).

### LOW-15: Categorical feature transform is O(n) Python loop
**File:** `src/openboost/_array.py:117-133`

**Status (2026-07-20): OPEN.** The per-row Python loop remains (unseen categories now at least map to `MISSING_BIN` with a warning instead of bin 0).

### LOW-16: `CustomDistribution` JAX gradients computed sample-by-sample
**File:** `src/openboost/_distributions.py:1582-1601`
No `jax.vmap`, kernels re-traced on every call.

**Status (2026-07-20): FIXED** (2026-07-20 distributions pass). The JAX path is `jax.vmap`-batched (with a per-sample fallback) and differentiates w.r.t. raw parameters.

### LOW-17: No CPU equivalent for categorical prediction
**File:** `_backends/_cpu.py` -- `predict_with_categorical_cpu` is absent.

**Status (2026-07-20): FIXED.** `predict_with_categorical_cpu` exists with a 64-bit bitset contract matching the GPU path.

---

## Testing Gaps

Status annotations added 2026-07-20 in the right-hand column.

| Area | Status |
|------|--------|
| `huber_gradient` | FIXED — covered in `tests/test_loss_correctness.py` |
| `NaturalBoostLogNormal` | FIXED — end-to-end coverage in `tests/test_integration.py`; gradient/NLL checks in `tests/test_distribution_gradients.py` |
| `NaturalBoostGamma` | FIXED — `tests/test_integration.py` + NLL-decrease fit test in `tests/test_distribution_gradients.py` |
| `NaturalBoostPoisson` | FIXED — `tests/test_integration.py` + NLL-decrease fit test in `tests/test_distribution_gradients.py` |
| `OpenBoostGAM` | FIXED — functional CPU tests in `tests/test_gam.py` |
| Sampling module (`goss_sample`, `MiniBatchIterator`, etc.) | FIXED — covered in `tests/test_large_scale.py` (no separate `test_sampling.py`) |
| GPU tests in CI | OPEN — `tests/modal_gpu_tests.py` exists but is manual; CI is still CPU-only |
| Loss integration tests | FIXED — `tests/test_loss_correctness.py` verifies gradients against finite differences and closed forms |
| `DistributionalGBDT` / `DART` | PARTIALLY FIXED — both models support callbacks/eval_set (incl. early stopping) in code; NaturalBoost callback behavior is exercised in `tests/test_distribution_gradients.py`, but DART-specific early-stopping tests are still missing |
| `MultiClassGradientBoosting` | OPEN — validation is implemented (MED-9) but has no dedicated tests |

## CI/CD Issues

Status annotations added 2026-07-20 in the right-hand column.

| Issue | Details |
|-------|---------|
| No GPU CI | OPEN — all CUDA code still untested in CI (Modal-based GPU tests are run manually) |
| No Windows CI | OPEN — matrix is still Ubuntu + macOS |
| `scipy` not declared as dependency | FIXED — `scipy>=1.10` is in `pyproject.toml` dependencies |
| Publish workflow has no test gate | FIXED — publish `needs: [test, bare-install]` in `.github/workflows/publish.yml` |
| PyPI classifier says "Beta" for 1.0.0rc1 | OPEN — classifier still `4 - Beta` while version is `1.0.0rc1` |

## Documentation Issues

Status annotations added 2026-07-20 in the right-hand column.

| Issue | Location |
|----------|----------|
| `compute_feature_importances` docs pass `model.trees_` instead of `model` | FIXED (2026-07-20 docs refresh) — `docs/quickstart.md`, `docs/user-guide/models/gradient-boosting.md`, `docs/migration/from-xgboost.md` now pass the fitted model |
| `sample()` shape comment is wrong | FIXED (2026-07-20 docs refresh) — `docs/quickstart.md` now states shape `(n_test, n_samples)` |
| Persistence docs show `pickle.load` with no security caveat | FIXED (2026-07-20 docs refresh) — `docs/user-guide/model-persistence.md` now carries an explicit untrusted-file warning |

---

## Summary Statistics

| Severity | Count |
|----------|-------|
| Critical (P0) | 8 |
| High (P1) | 12 |
| Medium (P2) | 21 |
| Low (P3) | 17 |
| **Total findings** | **58** |

### Status roll-up (2026-07-20)

| Severity | Fixed | Open |
|----------|-------|------|
| Critical (P0) | 8 (CRIT-5 by gating, kernel rewrite pending) | 0 |
| High (P1) | 10 | 2 (HIGH-1, HIGH-10) |
| Medium (P2) | 20 | 1 (MED-3) |
| Low (P3) | 14 | 3 (LOW-2, LOW-14, LOW-15) |
