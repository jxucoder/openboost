# MVP Implementation

## Time

2026-01-04

## Context

Need a working GPU gradient boosting implementation in pure Python (numba-cuda) with sklearn-style API. Target: 100K samples, 50 features, 20 trees, depth 6 in under 5 seconds.

## Original Plan

Follow project-0-mvp.md spec:
- Quantile binning to uint8
- Feature-major layout for coalesced access
- Level-order tree building (O(depth) kernel launches)
- Shared memory histograms
- Flat tree arrays for GPU prediction

## What Happened

### Implementation

Created 6 modules (~480 lines total):

| Module | Lines | Purpose |
|--------|-------|---------|
| `_kernels.py` | ~170 | All `@cuda.jit` at module level |
| `_boosting.py` | ~150 | GradientBoosting class |
| `_histogram.py` | ~40 | Histogram dispatch |
| `_split.py` | ~60 | Best split finding |
| `_array.py` | ~30 | Quantile binning |
| `_tree.py` | ~30 | Flat tree structure |

### Testing Challenges

1. **Modal image issues**: Started with `debian_slim` → missing CUDA runtime. Then `cuda:runtime` → missing NVVM compiler. Finally `cuda:devel` worked.

2. **Modal API changes**: `modal.Mount` deprecated in v1.0, switched to `image.add_local_dir()`.

### Benchmark Results (T4 GPU)

| Metric | OpenBoost | XGBoost GPU | Ratio |
|--------|-----------|-------------|-------|
| Train | 4.88s | 1.36s | 3.6x |
| Predict | 0.38s | 0.11s | 3.5x |
| MSE | 0.9938 | 0.9744 | ~same |

All success criteria passed.

## Lessons

1. **Numba overhead is real**: 3-4x vs XGBoost is consistent with expected "Python/Numba tax" for complex algorithms at this scale.

2. **Modal needs devel images**: For numba-cuda, always use `nvidia/cuda:*-devel-*` not `runtime`.

3. **JIT warmup matters**: First run includes compilation. Production would benefit from warmup.

4. **Grid size warnings are fine**: Early tree levels (1, 2, 4 nodes) trigger low-occupancy warnings but don't hurt overall performance.

## Future Plans

- Profile to find actual bottlenecks
- Consider histogram subtraction trick (compute sibling from parent)
- Explore larger datasets where GPU advantage grows

