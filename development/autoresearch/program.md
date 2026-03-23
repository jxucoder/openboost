# OpenBoost Autoresearch Program

You are an autonomous research agent improving the performance of **OpenBoost**, a GPU-accelerated gradient boosting library written in Python. Your goal: make training faster without breaking correctness.

## Objective

Reduce the total training time measured by `development/autoresearch/evaluate.py`. Each experiment runs a fixed benchmark (1M samples, 100 features, 200 trees, depth 8) and measures wall-clock time. Lower is better.

## Rules

1. **One change per experiment.** Make a single, focused optimization. Don't combine multiple ideas — isolate what works.
2. **Correctness is non-negotiable.** All tests must pass. If `evaluate.py` reports FAIL, discard the change immediately.
3. **Measure before and after.** Run `uv run python development/autoresearch/evaluate.py` before your change to establish baseline, then after to measure the delta.
4. **Commit improvements only.** If the score improves (lower time) AND tests pass, commit with a descriptive message. If not, `git checkout -- .` to discard.
5. **Don't modify test files or evaluation scripts.** Only modify source code under `src/openboost/`.
6. **Don't add dependencies.** Only use stdlib + numpy + numba + joblib (the existing deps).

## Workflow

```
1. Run: uv run python development/autoresearch/evaluate.py
   → Note the SCORE (training time in seconds)

2. Run: uv run python benchmarks/profile_loop.py --summarize
   → Read TOP BOTTLENECK, TARGET, and RECOMMENDATION

3. Read the target source file(s) identified by the profiler

4. Implement ONE optimization targeting the top bottleneck

5. Run: uv run python development/autoresearch/evaluate.py
   → If RESULT: PASS and score improved → commit
   → If RESULT: FAIL or score regressed → discard with: git checkout -- .

6. Repeat from step 1
```

## Mutable Files (what you CAN modify)

These are the optimization targets, ordered by typical impact:

### Tier 1: Core Kernels (highest impact)
- `src/openboost/_backends/_cpu.py` — Numba JIT CPU kernels (histogram building, split finding)
- `src/openboost/_backends/_cuda.py` — CUDA GPU kernels
- `src/openboost/_core/_primitives.py` — Histogram, split, partition orchestration
- `src/openboost/_core/_split.py` — Split evaluation logic

### Tier 2: Tree Building
- `src/openboost/_core/_growth.py` — Growth strategies (level-wise, leaf-wise, symmetric)
- `src/openboost/_core/_tree.py` — Tree fitting entry points
- `src/openboost/_core/_predict.py` — Prediction kernels

### Tier 3: Training Loop
- `src/openboost/_models/_boosting.py` — Main training loop, gradient/prediction updates
- `src/openboost/_loss.py` — Loss function implementations
- `src/openboost/_array.py` — BinnedArray data structure

## Immutable Files (do NOT modify)
- `development/autoresearch/evaluate.py` — Evaluation harness
- `benchmarks/*` — Benchmark scripts
- `tests/*` — Test suite
- `pyproject.toml` — Project configuration

## Optimization Strategies to Try

### CPU Kernel Optimizations
- **Loop reordering** for cache-friendly access patterns in histogram building
- **Feature batching** — process multiple features per inner loop iteration
- **Reducing branch mispredictions** in split evaluation
- **Removing redundant computations** in gradient/hessian accumulation
- **Tighter Numba `@njit` signatures** — explicit types reduce JIT overhead
- **Parallel chunking** — better work distribution in `prange` loops

### Algorithm Optimizations
- **Histogram subtraction trick** — compute child histogram from parent minus sibling
- **Skip empty bins** in split finding
- **Early termination** — stop split search when gain can't exceed current best
- **Lazy evaluation** — defer computation until actually needed
- **Memory pre-allocation** — avoid repeated array allocation in the training loop

### Training Loop Optimizations
- **Skip loss evaluation** when no early-stopping callback is registered
- **Fuse gradient + prediction update** into a single pass
- **Avoid CPU-GPU copies** in the prediction update path
- **Batch tree predictions** instead of per-tree traversal

### Data Structure Optimizations
- **Memory layout** — ensure contiguous access in hot loops
- **Reducing data copies** in BinnedArray operations
- **Compact node representation** to improve cache utilization

## What NOT to Do

- Don't change the public API (function signatures, class interfaces)
- Don't add approximate/lossy algorithms that change model output
- Don't modify binning logic (changes model numerics)
- Don't add multiprocessing/threading complexity without measuring
- Don't chase micro-optimizations in cold paths — focus on the top bottleneck
- Don't change random seeds or test tolerances to make tests pass
- Don't modify the evaluation harness or benchmarks

## Tips

- The profiler breaks training into phases: `histogram_build`, `split_find`, `partition`, `leaf_values`, `tree_overhead`, `grad_pred_loss`. Focus on whichever phase dominates.
- `grad_pred_loss` includes gradient computation, prediction updates, and loss evaluation. If this dominates, look at `_boosting.py`'s training loop and `_loss.py`.
- `tree_overhead` is Python-level overhead in the growth strategy. If this dominates, look at `_growth.py` for unnecessary object allocation or redundant computation.
- Run `uv run python benchmarks/profile_loop.py` (without `--summarize`) for a detailed per-phase table.
- Numba JIT functions have a warmup cost on first call. The evaluation harness handles warmup, so don't worry about it.
- Check `git log --oneline` to see what optimizations have already been tried.
