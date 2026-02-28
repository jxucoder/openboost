# OpenBoost Code Review

**Date:** 2026-02-28
**Scope:** Full source code review of `src/openboost/` (40 source files)

---

## Summary

| Severity | Count |
|----------|-------|
| Critical | 5     |
| High     | 12    |
| Medium   | 12    |
| Low      | 4     |
| **Total**| **33**|

The most impactful issues fall into a few themes:
1. **Train/predict inconsistencies** — different data layouts, binning logic, or missing-value handling between training and inference
2. **Loss/metric bugs** — early stopping and validation always use MSE regardless of the configured loss function
3. **GPU kernel correctness** — race conditions, uninitialized kernels, missing-value handling omitted
4. **Distributed/multi-GPU breakage** — invalid API calls, race conditions, incorrect histogram construction

---

## Critical Issues

### C1. Import of non-existent `quantile_bin` crashes the boosting module
**File:** `_boosting.py:6`

```python
from ._array import quantile_bin  # Does not exist — actual name is _quantile_bin
```

The function in `_array.py` is `_quantile_bin` (private, with leading underscore). This `ImportError` makes the entire `GradientBoosting` class in `_boosting.py` unusable at import time.

---

### C2. Training data layout mismatch — kernels read garbage during fit
**File:** `_boosting.py:55` (fit) vs `_boosting.py:192` (predict)

During `fit()`, data is binned via `quantile_bin(X)` → `_quantile_bin` → `_bin_features`, which returns **(n_samples, n_features)** order. But the CUDA kernels (`histogram_kernel`, `predict_kernel`, `update_sample_nodes_kernel`) all index as `X_binned[feature, sample]`, expecting **(n_features, n_samples)** order. The data is never transposed before being sent to the GPU.

In `predict()`, the array is correctly built as `(n_features, n_samples)`, so predictions on a trained model will also be wrong since the trees were fitted on transposed data.

**Impact:** Training produces garbage models — every feature lookup reads from the wrong location.

---

### C3. GPU split-finding race condition silently corrupts tree splits
**File:** `_backends/_cuda.py:1981-1989`

```python
cuda.atomic.max(best_gains_out, config_idx, best_gain)
# Race: another thread can win between atomic.max and the equality check
if best_gains_out[config_idx] == best_gain:
    best_features_out[config_idx] = feature_idx   # Non-atomic write
    best_bins_out[config_idx] = best_bin           # Non-atomic write
```

The code acknowledges the race condition in a comment. Between `atomic.max` and the equality check, another block can overwrite the gain. The writes to `best_features_out` and `best_bins_out` are non-atomic, so the final result can have a gain from one feature but feature/bin indices from a different feature.

**Impact:** Trees may split on the wrong feature/threshold, silently degrading model quality.

---

### C4. Leaf-wise growth mutates the shared `config` object
**File:** `_core/_growth.py:627-628`

```python
if config.max_leaves is None:
    config.max_leaves = 2**config.max_depth
```

The `GrowthConfig` dataclass is mutated in place. If the same config is reused across trees in boosting (the common case), the first call permanently sets `max_leaves`, and subsequent changes to `max_depth` will not update it.

**Impact:** Silently stale configuration across trees when configs are reused.

---

### C5. Multi-GPU `ob.array()` called with invalid `bin_edges` kwarg — crashes at runtime
**File:** `_distributed/_multigpu.py:91`

```python
self.X_binned = ob.array(X_shard, n_bins=n_bins, bin_edges=bin_edges)
```

The `ob.array()` function signature is `array(X, n_bins=256, *, categorical_features=None, device=None)` — it does not accept a `bin_edges` parameter. This raises `TypeError` at runtime, making multi-GPU training completely broken when using pre-computed bin edges (the normal path in `MultiGPUContext.setup`).

---

## High Severity Issues

### H1. `_compute_loss` ignores the loss function — always computes MSE
**File:** `_training.py:228-237`

```python
def _compute_loss(pred: NDArray, y: NDArray, loss_fn) -> float:
    diff = pred - y
    return float(np.mean(diff ** 2))  # Always MSE, ignores loss_fn
```

The `loss_fn` parameter is accepted but never used. Early stopping, validation metrics, and logged losses are always MSE regardless of the configured loss.

**Impact:** Early stopping terminates based on the wrong metric. For classification (logloss), MSE on logits is meaningless.

---

### H2. Hardcoded MSE for train/val loss in `_models/_boosting.py`
**File:** `_models/_boosting.py:673, 678`

```python
state.train_loss = float(np.mean((pred_cpu - y) ** 2))  # Always MSE
state.val_loss = float(np.mean((val_pred - y_val) ** 2))  # Always MSE
```

Same issue as H1 but in the GPU training path. Both `_fit_gpu` and `_fit_cpu` hardcode MSE for the loss reported to callbacks.

---

### H3. Training predictions start at zero, ignoring initial prediction (bias)
**File:** `_training.py:139`

```python
pred = np.zeros(n_samples, dtype=np.float32)
```

Most models set `initial_prediction_` (e.g., `mean(y)` for regression, log-odds for classification). The training loop doesn't incorporate it, so the first tree wastes capacity learning the difference from zero.

**File:** `_training.py:251` — Same issue for validation predictions in `_default_predict`.

---

### H4. Predict-time binning inconsistent with train-time binning
**File:** `_boosting.py:192-194`

```python
X_binned[f] = np.digitize(X[:, f], edges[1:-1], right=False).astype(np.uint8)
```

During `fit()`, binning uses `np.digitize(col, edges)` with full edges. During `predict()`, edges are sliced with `edges[1:-1]`, removing the first and last. Boundary samples get assigned to different bins, causing incorrect tree traversal.

---

### H5. Validation data re-binned every round with wrong edges
**File:** `_training.py:247-248`

```python
X_binned = array(X, n_bins=256)  # Called every round, creates new bin edges from val data
```

`_default_predict` is called every training round. Raw validation data is re-binned from scratch each time with **new** edges computed from the validation set (not the training edges). This causes: (1) O(n * features) overhead per round, and (2) validation splits that don't match training splits.

---

### H6. Incorrect child histogram construction in multi-GPU tree building
**File:** `_models/_boosting.py:479-489`

```python
for f in range(n_features):
    left_hist_grad[f, :split.threshold + 1] = h_grad[f, :split.threshold + 1]
```

After a split, the code copies bins `<= threshold` for **all features** into the left histogram. But the threshold only applies to the split feature. For non-split features, which bins go left vs. right depends on sample assignment, not bin value. This produces incorrect gradient statistics for all subsequent splits in child nodes.

---

### H7. DART normalization weight decay compounds across rounds
**File:** `_models/_dart.py:162-176`

Tree weights include the learning rate (`weight * self.learning_rate`). DART normalization multiplies existing weights by `k/(k+1)` each round. Since the learning rate is baked in, repeated normalization causes older trees' effective contribution to decay cumulatively far more than the DART paper intends.

**Impact:** Older trees become negligible, defeating DART's purpose of preventing later trees from dominating.

---

### H8. `LinearLeafTree` uses floating-point comparison for leaf routing
**File:** `_models/_linear_leaf.py:93, 285`

Leaf IDs are `float(v)` dict keys from tree predictions. At predict time, `if leaf_pred in self.leaf_ids` uses exact `==` on floats. Different floating-point rounding between train and predict binning can cause mismatches, silently routing samples to the wrong leaf (fallback: `leaf_idx = 0`).

---

### H9. `NegativeBinomial.nll` has incorrect sign computation
**File:** `_distributions.py:1370-1374`

The expression computes the combinatorial part with NLL signs but the exponential part (`r*log(p) + y*log(1-p)`) with log-PMF signs. Multiplying by `-1` doesn't fix both. The result is neither the NLL nor the log-PMF.

**Impact:** Distributional model evaluation and scoring for NegBin models is wrong.

---

### H10. `_count_nodes` function is broken in level-wise growth
**File:** `_core/_growth.py:582-593`

```python
return max(left_children[parent], left_children[parent] + 1, i) + 1
```

`left_children[parent]` is a child node index, not a count. The expression `max(child, child+1, i)` doesn't correctly count all tree nodes. This truncates or oversizes the tree arrays, causing wrong predictions or out-of-bounds access.

**Duplicate:** Same broken logic exists at `_distributed/_tree.py:153-160`.

---

### H11. Distributed tree building: partition not awaited before next level
**File:** `_distributed/_tree.py:97-100`

```python
partition_refs = [w.partition_samples.remote(splits) for w in workers]
active_nodes = new_active_nodes  # Immediately continues without ray.get()
```

`partition_samples` is dispatched but never awaited. The next iteration's `compute_histograms` may execute before sample-to-node assignments are updated, building histograms on stale assignments.

---

### H12. GPU prediction ignores missing-value direction and categorical splits
**File:** `_core/_growth.py:338-350`

`_predict_standard_gpu` calls the basic `predict_cuda` without passing `missing_go_left`, `is_categorical_split`, or `cat_bitsets`. Missing values always go right on GPU but may have been trained to go left. Categorical splits are ignored entirely.

**Impact:** GPU and CPU predictions disagree whenever trees have learned missing-goes-left or categorical splits.

---

## Medium Severity Issues

### M1. `LearningRateScheduler` callback has no effect on training
**File:** `_training.py:190` vs `_callbacks.py:332`

The scheduler modifies `state.model.learning_rate`, but the training loop reads `config.learning_rate` (never updated). Learning rate scheduling silently does nothing.

---

### M2. Bin index collision with `MISSING_BIN` (255)
**File:** `_array.py:210-211`

When `n_bins=256` (the default) and a feature has exactly 255 unique quantile values, `np.digitize` can return 255, colliding with `MISSING_BIN`. Non-missing values would be misclassified as missing.

---

### M3. Categorical `transform()` uses `int(val)` but map keys are `np.float64`
**File:** `_array.py:116-122`

```python
key = int(val)           # Python int
cat_map.get(key, 0)      # Keys are np.float64 from training
```

Dict lookup with `int` key against `np.float64` keys fails to match. All categories silently map to bin 0 during transform, destroying categorical information.

---

### M4. `EarlyStopping` snapshots trees via shallow copy
**File:** `_callbacks.py:182`

```python
self._best_trees = [t for t in state.model.trees_]
```

If tree objects are later mutated (e.g., pruned), the "best" snapshot is corrupted. Should deep-copy.

---

### M5. `_predict_tree_add_kernel` may be `None` when called
**File:** `_core/_predict.py:131-134`

The kernel is initialized to `None` and only set during module import if CUDA is available. The function `predict_tree_add_gpu` uses it directly without calling `_get_predict_tree_add_kernel()` to ensure it's compiled. Same pattern affects all GPU kernels in `_loss.py` (MSE, logloss, huber, etc.).

---

### M6. CUDA kernels recompiled on every call in `_predict.py`
**File:** `_predict.py:65-92`

`_fill_cuda` and `_add_inplace_cuda` define `@cuda.jit` kernels inside the function body, creating new function objects each call. Unlike module-level kernels, these may trigger recompilation per invocation — significant overhead during prediction with many trees.

---

### M7. `n_samples = X.shape[1]` assumes feature-major layout for raw arrays
**File:** `_predict.py:42`

For non-`BinnedArray` input, the code assumes `(n_features, n_samples)` layout, but standard convention is `(n_samples, n_features)`. Users passing standard numpy arrays get silent wrong results.

---

### M8. `MultiClassGradientBoosting.fit()` never sets `n_features_in_`
**File:** `_models/_boosting.py:986-1013`

The attribute stays at its default of `0`. Direct users (not through sklearn wrapper) will see incorrect feature count.

---

### M9. `OpenBoostClassifier` ignores `sample_weight`, `eval_set`, and `callbacks` for multi-class
**File:** `_models/_sklearn.py:526-530`

The multi-class path calls `self.booster_.fit(X, y_encoded)` without forwarding `sample_weight`, `eval_set`, or callbacks. Early stopping is silently disabled for multi-class problems.

---

### M10. `_is_diagonal_fisher` only checks the first sample
**File:** `_distributions.py:189-196`

Only inspects `F[0]` to decide if the Fisher matrix is diagonal. If parameters vary across samples, the diagonal fast-path could be incorrectly used for the whole batch.

---

### M11. `LinearLeafGBDT._fit_weighted_ridge` creates O(n^2) matrix
**File:** `_models/_linear_leaf.py:386`

```python
W = np.diag(weights)  # (n_samples, n_samples) dense matrix
```

For a leaf with 10,000 samples, this creates a ~800MB matrix. The correct approach: `X.T @ (weights[:, None] * X)`.

---

### M12. Symmetric GPU prediction reads wrong indices (missing leaf offset)
**File:** `_core/_growth.py:377-387`

The GPU kernel `predict_symmetric_cuda` indexes leaf values starting from 0, but leaf values are stored starting at index `2^depth - 1`. The CPU path correctly adds this offset. GPU predictions return internal node values (zeros) instead of leaf values.

---

## Low Severity Issues

### L1. Quantile gradient docstring has swapped descriptions
**File:** `_loss.py:429-432` — "pred > y" labeled as "under-prediction" (should be over-prediction).

### L2. `@staticmethod` on module-level function (dead code)
**File:** `_models/_boosting.py:891` — `@staticmethod` is meaningless outside a class body.

### L3. `_sum_histogram` sums all features instead of one
**File:** `_core/_split.py:136-140` — Returns `n_features * actual_total`. Only hit if callers don't provide totals (currently they always do).

### L4. CPU uses float64 vs GPU uses float32 for split finding
**File:** `_backends/_cpu.py:298` vs `_backends/_cuda.py:1373` — Can produce different "best split" results, hurting reproducibility.
