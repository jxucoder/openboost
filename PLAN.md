# Fix Plan for OpenBoost Code Review Issues

Fixes are grouped into 8 phases by theme. Each phase targets related issues to
minimize churn and ensure correctness. Within each phase, issues are ordered by
severity.

---

## Phase 1: Fix `_boosting.py` (legacy boosting module) — C1, C2, H4

These are the highest-severity issues and block the entire legacy GradientBoosting
class from working.

### C1 — Fix broken import (`_boosting.py:6`)
- Change `from ._array import quantile_bin` → `from ._array import _quantile_bin`
- Update the call site at line 55: `quantile_bin(X)` → `_quantile_bin(X)`

### C2 — Fix data layout mismatch (`_boosting.py:55`)
- After `X_binned, self.bin_edges_ = _quantile_bin(X)`, transpose the result:
  `X_binned = X_binned.T` so it becomes `(n_features, n_samples)` matching the
  GPU kernel expectations (`X_binned[feature, sample]`).

### H4 — Fix predict-time binning (`_boosting.py:192-194`)
- Change `edges[1:-1]` to `edges` to use the full edges array, matching the
  binning logic in `_quantile_bin` / `_bin_features`.
- Verify that `np.digitize(X[:, f], edges, right=False)` with full edges produces
  the same bin indices as training.

---

## Phase 2: Fix loss computation and training loop — H1, H2, H3, H5, M1

These all relate to the training infrastructure producing wrong metrics and
predictions.

### H1 — Make `_compute_loss` use the actual loss function (`_training.py:228-237`)
- Replace the hardcoded MSE with a proper dispatch. The `loss_fn` parameter has
  signature `(pred, y) -> (grad, hess)` which doesn't directly give a scalar loss.
- Add a `_loss_value` helper that computes the actual loss:
  - Use `loss_fn(pred, y)` to get `(grad, hess)`.
  - For loss value: compute `0.5 * mean(grad^2 / hess)` as a second-order
    approximation, OR add a `loss_value` method to loss functions.
  - Simpler approach: check if `loss_fn` has a `loss` attribute or callable. If
    not available, fall back to `mean((pred - y)^2)` but only for MSE.
  - Best approach: add an optional `loss_value_fn` parameter to `TrainingConfig`
    that takes `(pred, y) -> float`. When provided, use it. The models that set up
    the training config can pass the appropriate loss metric.

### H2 — Fix hardcoded MSE in `_models/_boosting.py` GPU/CPU training paths
  (`_models/_boosting.py:673, 678, ~793, ~798`)
- Same issue but in the model-level training loops. Replace:
  ```python
  state.train_loss = float(np.mean((pred_cpu - y) ** 2))
  ```
  with a call to the model's actual loss function's value method.
- The `GradientBoosting` dataclass has a `loss` attribute (string). Map it to a
  loss value function:
  - `'mse'` → `mean((pred - y)^2)`
  - `'logloss'` → `mean(-y*log(sigmoid(pred)) - (1-y)*log(1-sigmoid(pred)))`
  - `'mae'` → `mean(|pred - y|)`
  - `'huber'` → standard huber loss
  - `'quantile'` → quantile loss
  - `'poisson'` → poisson deviance
  - `'gamma'` → gamma deviance
  - `'tweedie'` → tweedie deviance
- Add a `_compute_loss_value(self, pred, y)` method to `GradientBoosting` that
  dispatches based on `self.loss`.

### H3 — Initialize predictions with model's initial prediction (`_training.py:139, 251`)
- Line 139: Change `pred = np.zeros(...)` to:
  ```python
  init_pred = getattr(model, 'initial_prediction_', 0.0)
  pred = np.full(n_samples, init_pred, dtype=np.float32)
  ```
- Line 251 in `_default_predict`: Same fix — start from `initial_prediction_`
  instead of zeros:
  ```python
  init_pred = getattr(model, 'initial_prediction_', 0.0)
  pred = np.full(n_samples, init_pred, dtype=np.float32)
  ```

### H5 — Cache validation binning in `_default_predict` (`_training.py:247-248`)
- The re-binning every round is the symptom. The fix: `_default_predict` should
  use the model's existing `BinnedArray`/bin edges to transform validation data,
  not create new bin edges.
- Change `array(X, n_bins=256)` to use the model's stored binned array's
  `transform()` method:
  ```python
  if hasattr(model, 'X_binned_') and model.X_binned_ is not None:
      X_binned = model.X_binned_.transform(X)
  else:
      X_binned = array(X, n_bins=256)
  ```
- Additionally, cache the binned validation data. Add a `_val_binned_cache`
  dict attribute or parameter to avoid re-binning across rounds. The simplest
  approach: accept an optional pre-binned `X_val_binned` in the training loop
  and bin once before the loop starts.

### M1 — Fix `LearningRateScheduler` reading from config (`_training.py:190`)
- In `run_training_loop`, after the callback's `on_round_begin` (which may update
  `model.learning_rate`), read the learning rate from the model instead of config:
  ```python
  lr = getattr(model, 'learning_rate', config.learning_rate)
  pred = pred + lr * tree_pred
  ```

---

## Phase 3: Fix GPU kernel initialization pattern — M5, M6, L2

Multiple files have the same anti-pattern: kernel initialized to `None` at module
level, with a `_ensure_*` function that exists but is never called.

### M5 — Use lazy init in all GPU kernel call sites
Files affected:
- `_core/_predict.py:131` — call `_get_predict_tree_add_kernel()` if `None`
- `_loss.py` — for every `_*_gradient_kernel`, call `_ensure_*_kernel()` before use
- `_models/_boosting.py:888` — call init before `_fill_zeros_kernel` usage

Pattern for each:
```python
# Before using the kernel:
global _some_kernel
if _some_kernel is None:
    _some_kernel = _get_some_kernel()
```

Or simpler: just call the `_ensure_*` function at the top of each GPU function.

### M6 — Move CUDA kernels to module level in `_predict.py`
- Extract `_fill_cuda` and `_add_inplace_cuda`'s inner `@cuda.jit` kernels to
  module-level definitions (same lazy-init pattern as other kernels).

### L2 — Remove `@staticmethod` on module-level function (`_models/_boosting.py:891`)
- Remove the `@staticmethod` decorator from `_get_fill_zeros_kernel()`.
- This is dead code since the kernel is defined inline at lines 910-916 instead.
  Remove the entire `_get_fill_zeros_kernel` function.

---

## Phase 4: Fix GPU split-finding race condition — C3

### C3 — Eliminate race in batch split finding (`_backends/_cuda.py:1981-1989`)
The proper fix for CUDA is to encode `(gain, feature, bin)` into a single 64-bit
value that can be atomically compared.

- Encode `gain` as the upper 32 bits (as a float32 reinterpreted as uint32) and
  pack `feature` (16 bits) + `bin` (16 bits) into the lower 32 bits.
- Use `cuda.atomic.max` on the combined 64-bit value.
- After the kernel, decode the winning `(feature, bin)` from the lower 32 bits.

Alternative simpler fix (less optimal but correct):
- Run a two-pass approach:
  1. First kernel: each block finds its local best gain per feature and writes
     `best_gains_per_feature[config_idx, feature_idx]`.
  2. Second kernel (or host code): scan `best_gains_per_feature` to find the
     global best for each config.
- This eliminates the cross-block race entirely.

---

## Phase 5: Fix distributed/multi-GPU — C5, H6, H10, H11

### C5 — Add `bin_edges` support to `ob.array()` (`_distributed/_multigpu.py:91`)
Two options:
- **Option A**: Add a `bin_edges` parameter to `array()` in `_array.py` that uses
  pre-computed edges instead of computing new ones. Then `BinnedArray.transform()`
  already handles this — so the fix is to compute bin edges on the full dataset
  first, then use `transform()` on each shard.
- **Option B** (simpler): Replace line 91 with:
  ```python
  # Create a BinnedArray from the global bin_edges, then transform the shard
  from .._array import BinnedArray
  template = BinnedArray.__new__(BinnedArray)
  template.bin_edges = bin_edges
  self.X_binned = template.transform(X_shard)
  ```
- Best approach: Create a standalone `bin_with_edges(X, bin_edges)` function in
  `_array.py` that bins data using pre-computed edges. Call it from `_multigpu.py`.

### H6 — Fix child histogram construction (`_models/_boosting.py:479-489`)
- The current code applies the split threshold to ALL features. The correct
  approach for histogram-based GBDT: after splitting, each child's histogram
  for non-split features must be computed from sample assignments, not bin
  thresholds.
- Since this is in the multi-GPU path that builds histograms by subtraction
  without re-scanning samples, the proper fix is:
  - For the **split feature**: partition bins by threshold (current logic, correct
    for the split feature only).
  - For **non-split features**: you cannot partition histograms without knowing
    which samples went left vs right. The histogram subtraction trick only works
    if you have the parent histogram and ONE child's histogram (computed from
    re-scanning samples in that child).
  - Fix: After the split, re-scan samples to build the smaller child's histogram,
    then derive the larger child by subtraction from the parent. This requires
    access to sample-to-node assignments.
  - Alternative: remove this code path and use the correct histogram building from
    `_core/_primitives.py` which re-scans samples per node.

### H10 — Fix `_count_nodes` (`_core/_growth.py:582-593`, `_distributed/_tree.py:153-160`)
- Replace with a correct implementation that walks the tree:
  ```python
  def _count_nodes(left_children) -> int:
      count = 0
      for i in range(len(left_children)):
          # Node is valid if it's the root or its parent split (has a valid left child)
          if i == 0 or left_children[(i - 1) // 2] != -1:
              count = i + 1  # Track highest valid index + 1
      return max(count, 1)
  ```
- Extract to a shared utility to eliminate the duplicate in `_distributed/_tree.py`.

### H11 — Await partition refs before next level (`_distributed/_tree.py:97-100`)
- Add `ray.get(partition_refs)` after dispatching:
  ```python
  if splits:
      partition_refs = [w.partition_samples.remote(splits) for w in workers]
      ray.get(partition_refs)  # Wait for partition to complete
  ```

---

## Phase 6: Fix model-specific bugs — H7, H8, H9, H12, M8, M9, M11, M12

### H7 — Fix DART weight normalization (`_models/_dart.py:162-173`)
- Store learning rate separately from tree weights. Change line 173:
  ```python
  self.tree_weights_.append(tree_weight)  # Don't bake in learning_rate
  ```
- Apply learning rate during prediction instead (in `_predict_internal`):
  ```python
  pred += self.learning_rate * weight * tree_pred
  ```

### H8 — Fix float-key dict in LinearLeafTree (`_models/_linear_leaf.py:93, 285`)
- Use integer leaf indices instead of float prediction values as keys.
- Change line 285 to use a rounded/quantized key or, better, assign integer IDs:
  ```python
  leaf_ids = {i: i for i, v in enumerate(unique_leaves)}
  ```
- At predict time, map continuous predictions to the nearest leaf using
  `np.searchsorted` or `np.argmin(|pred - unique_leaves|)` instead of exact
  dict lookup.

### H9 — Fix NegativeBinomial NLL (`_distributions.py:1370-1374`)
- The correct NLL is:
  ```python
  nll = -(gammaln(y_int + r) - gammaln(r) - gammaln(y_int + 1)
          + r * np.log(r / (r + μ))
          + y_int * np.log(μ / (r + μ)))
  ```
- Fix by swapping the signs in the gammaln terms:
  ```python
  return -(
      gammaln(y_int + r) - gammaln(r) - gammaln(y_int + 1)
      + r * np.log(r / (r + μ))
      + y_int * np.log(μ / (r + μ))
  )
  ```

### H12 — Pass missing/categorical info to GPU prediction (`_core/_growth.py:338-350`)
- Update `_predict_standard_gpu` to pass `missing_go_left`, `is_categorical_split`,
  and `cat_bitsets` to `predict_cuda`.
- This requires `predict_cuda` to have variants that handle these (check if
  `_backends/_cuda.py` already has `predict_with_missing_cuda` or similar). If not,
  use the categorical prediction kernel (`_predict_with_categorical_kernel`) when
  categorical splits exist, or the missing-aware kernel when missing directions
  exist.

### M8 — Set `n_features_in_` in MultiClass fit (`_models/_boosting.py:1013`)
- Add after line 1013:
  ```python
  self.n_features_in_ = self.X_binned_.n_features
  ```

### M9 — Forward kwargs in multi-class sklearn path (`_models/_sklearn.py:528-530`)
- Pass `sample_weight`, `eval_set`, and `callbacks` to multi-class fit.
- If `MultiClassGradientBoosting.fit()` doesn't accept these yet, add the
  parameters and wire them through.
- At minimum, emit a warning if these are provided but unsupported.

### M11 — Fix O(n^2) weighted ridge (`_models/_linear_leaf.py:386`)
- Replace `W = np.diag(weights)` and `X_aug.T @ W @ X_aug` with:
  ```python
  XtWX = X_aug.T @ (weights[:, None] * X_aug) + reg_matrix
  ```
- This is O(n * p^2) instead of O(n^2).

### M12 — Fix symmetric GPU prediction offset (`_core/_growth.py:377-387`)
- Pass the leaf offset to the GPU kernel, or extract only the leaf values before
  sending to GPU:
  ```python
  leaf_start = 2**self.depth - 1
  leaf_values = self.values[leaf_start:leaf_start + 2**self.depth]
  return predict_symmetric_cuda(binned, self.level_features,
      self.level_thresholds.astype(np.uint8), leaf_values, self.depth)
  ```

---

## Phase 7: Fix remaining medium issues — M2, M3, M4, M7, M10

### M2 — Fix bin index / MISSING_BIN collision (`_array.py:210-211`)
- Change cap from `n_bins > 255` to `n_bins > 254`:
  ```python
  if n_bins > 254:
      n_bins = 254  # Max 254 bins (0-253), leaving 255 for MISSING_BIN
  ```
- Also clip digitize output to `min(len(edges), 254)`.

### M3 — Fix categorical transform key type (`_array.py:116-122`)
- Ensure category map keys match transform lookup type. Change either:
  - Build category map with `int` keys: `{int(v): i for i, v in enumerate(unique_vals)}`
  - Or use `np.float64(val)` in transform instead of `int(val)`.
- Best: use `int` keys in the map AND `int(val)` in lookup (both sides).

### M4 — Deep-copy trees in EarlyStopping (`_callbacks.py:182`)
- Change to:
  ```python
  import copy
  self._best_trees = copy.deepcopy(state.model.trees_)
  ```

### M7 — Fix `n_samples` assumption in predict_ensemble (`_predict.py:42`)
- Document that non-BinnedArray input must be feature-major `(n_features, n_samples)`.
- Or better: detect and handle sample-major format:
  ```python
  # Convention: non-BinnedArray assumed to be feature-major (n_features, n_samples)
  n_samples = X.shape[1]
  ```
- Since the codebase consistently uses feature-major for GPU data, adding a clear
  assertion/docstring is sufficient.

### M10 — Check more samples in `_is_diagonal_fisher` (`_distributions.py:189-196`)
- Sample a few indices (e.g., min(10, n_samples)) and check all of them:
  ```python
  check_indices = np.linspace(0, len(F)-1, min(10, len(F)), dtype=int)
  for idx in check_indices:
      off_diag = np.sum(np.abs(F[idx])) - np.sum(np.abs(np.diag(F[idx])))
      if off_diag >= 1e-8:
          return False
  return True
  ```

---

## Phase 8: Fix low severity issues — L1, L3, L4

### L1 — Fix quantile docstring (`_loss.py:429-432`)
- Swap "under-prediction" and "over-prediction" labels.

### L3 — Fix `_sum_histogram` (`_core/_split.py:136-140`)
- Sum only one feature's bins (feature 0):
  ```python
  return float(np.sum(hist[0]))
  ```

### L4 — Document CPU/GPU precision difference
- Add a comment in `_backends/_cpu.py` noting the float64 choice and the
  implication for reproducibility vs GPU.

---

## Implementation Order

| Order | Phase | Issues | Risk | Files touched |
|-------|-------|--------|------|---------------|
| 1     | 1     | C1, C2, H4 | Low — isolated file | `_boosting.py` |
| 2     | 7     | M2, M3, M4, M7, M10 | Low — small targeted fixes | `_array.py`, `_callbacks.py`, `_predict.py`, `_distributions.py` |
| 3     | 2     | H1, H2, H3, H5, M1 | Medium — training infra | `_training.py`, `_models/_boosting.py` |
| 4     | 3     | M5, M6, L2 | Low — kernel init pattern | `_core/_predict.py`, `_loss.py`, `_models/_boosting.py`, `_predict.py` |
| 5     | 6     | H7-H12, M8, M9, M11, M12 | Medium — multiple models | `_models/_dart.py`, `_linear_leaf.py`, `_sklearn.py`, `_distributions.py`, `_core/_growth.py` |
| 6     | 8     | L1, L3, L4 | Low — docs/comments | `_loss.py`, `_core/_split.py`, `_backends/_cpu.py` |
| 7     | 4     | C3 | High — GPU kernel change | `_backends/_cuda.py` |
| 8     | 5     | C5, H6, H10, H11 | High — distributed paths | `_distributed/_multigpu.py`, `_distributed/_tree.py`, `_core/_growth.py`, `_array.py` |
