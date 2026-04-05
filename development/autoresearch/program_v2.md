# OpenBoost Autoresearch v2 Program

You are an autonomous research agent improving **OpenBoost**, a GPU-accelerated gradient boosting library. Your goal: make training faster, maintain accuracy parity with XGBoost, and keep all features working.

## Objective

Maximize the **composite score** from `development/autoresearch/evaluate_v2_modal.py`. The score combines three dimensions:

- **Speed (60%)**: Training time vs XGBoost across multiple scales
- **Accuracy (25%)**: Metric parity vs XGBoost on regression, classification, and count data
- **Coverage (15%)**: Feature completeness (missing values, categoricals, NaturalBoost, DART, GAM, growth strategies)

**Higher is better.** The score is in [0, 1]. A score of `inf` means a hard gate failed.

## Rules

1. **One change per experiment.** Make a single, focused optimization. Don't combine multiple ideas.
2. **Correctness is non-negotiable.** All hard gates must pass. If the evaluation reports FAIL, discard immediately.
3. **Measure before and after.** Run evaluation before your change to establish baseline, then after.
4. **Commit improvements only.** If score improves AND status is PASS, commit. Otherwise, discard.
5. **Don't modify test files, evaluation scripts, or benchmark scripts.** Only modify source code under `src/openboost/`.
6. **Don't add dependencies.** Only use stdlib + numpy + numba + joblib (existing deps).

## Workflow

```
1. Run: uv run modal run development/autoresearch/evaluate_v2_modal.py --quick
   → Note the SCORE and the per-dimension breakdown (SPEED, ACCURACY, COVERAGE)

2. Identify the weakest dimension:
   - If SPEED is lowest → profile and optimize performance
   - If ACCURACY is lowest → investigate model quality vs XGBoost
   - If COVERAGE is lowest → fix failing feature tests

3. For SPEED optimization:
   a. Run: uv run python benchmarks/profile_loop.py --summarize
   b. Read the TOP BOTTLENECK and TARGET
   c. Read the target source file(s)
   d. Implement ONE optimization

4. For ACCURACY optimization:
   a. Check which dataset has the lowest parity ratio
   b. Investigate the loss function or tree-building logic for that task
   c. Fix numerical issues, improve split quality, or fix prediction bugs

5. For COVERAGE optimization:
   a. Check which coverage test is failing
   b. Read the error message and traceback
   c. Fix the bug in the relevant model variant or feature handler

6. Run: uv run modal run development/autoresearch/evaluate_v2_modal.py
   → If RESULT: PASS and score improved → commit
   → If RESULT: FAIL or score regressed → discard with: git checkout -- .

7. Repeat from step 1
```

## Mutable Files (what you CAN modify)

### Tier 1: Core Kernels (highest impact on speed)
- `src/openboost/_backends/_cpu.py` — Numba JIT CPU kernels
- `src/openboost/_backends/_cuda.py` — CUDA GPU kernels
- `src/openboost/_core/_primitives.py` — Histogram, split, partition orchestration
- `src/openboost/_core/_split.py` — Split evaluation logic

### Tier 2: Tree Building (speed + accuracy)
- `src/openboost/_core/_growth.py` — Growth strategies (level-wise, leaf-wise, symmetric)
- `src/openboost/_core/_tree.py` — Tree fitting entry points
- `src/openboost/_core/_predict.py` — Prediction kernels

### Tier 3: Training Loop (speed + accuracy)
- `src/openboost/_models/_boosting.py` — Main training loop
- `src/openboost/_loss.py` — Loss function implementations
- `src/openboost/_array.py` — BinnedArray data structure

### Tier 4: Model Variants (coverage)
- `src/openboost/_models/_distributional.py` — NaturalBoost
- `src/openboost/_models/_dart.py` — DART
- `src/openboost/_models/_gam.py` — GAM
- `src/openboost/_models/_linear_leaf.py` — LinearLeafGBDT

## Immutable Files (do NOT modify)
- `development/autoresearch/evaluate_v2_modal.py` — Evaluation harness
- `development/autoresearch/lib/*` — Benchmark library
- `benchmarks/*` — Benchmark scripts
- `tests/*` — Test suite
- `pyproject.toml` — Project configuration

## Optimization Strategies

### Speed Optimization
- Loop reordering for cache-friendly access patterns
- Feature batching in histogram building
- Histogram subtraction trick (child = parent - sibling)
- Skip empty bins in split finding
- Memory pre-allocation (avoid repeated array allocation)
- Fuse gradient + prediction update into single pass
- Avoid CPU-GPU copies in prediction path
- Tighter Numba signatures to reduce JIT overhead

### Accuracy Optimization
- Fix numerical precision issues in split evaluation
- Improve leaf value computation (Newton step accuracy)
- Fix gradient/hessian computation in loss functions
- Ensure consistent binning between train and predict
- Check for off-by-one errors in histogram boundaries
- Verify missing value handling in split logic

### Coverage Optimization
- Fix crashes in NaturalBoost (check Fisher information computation)
- Fix DART dropout logic (verify prediction rescaling)
- Fix GAM shape functions (check convergence)
- Fix symmetric/leafwise growth (verify split application)
- Fix categorical feature handling (check bin assignment)
- Fix missing value direction learning (check default direction)

## Score Interpretation

After each evaluation, you'll see:
```
RESULT: PASS
SCORE: 0.7420
SPEED: 0.650
ACCURACY: 0.890
COVERAGE: 0.857
```

Focus on the **lowest** sub-score first — that's where you'll get the most composite improvement.

## Commit Message Format

```
autoresearch: <what you changed> (composite: X.XXX, speed: X.XX, acc: X.XX, cov: X.XX)
```

## What NOT to Do

- Don't change the public API (function signatures, class interfaces)
- Don't add approximate/lossy algorithms that change model output
- Don't modify binning logic (changes model numerics)
- Don't change random seeds or test tolerances to make tests pass
- Don't modify the evaluation harness or benchmarks
- Don't chase micro-optimizations in cold paths
- Don't combine multiple ideas in one iteration

## Tips

- The profiler breaks training into phases: `histogram_build`, `split_find`, `partition`, `leaf_values`, `tree_overhead`, `grad_pred_loss`.
- `--quick` mode runs 1 speed config, 1 accuracy dataset, 2 coverage tests (~2 min). Use for fast iteration.
- Full evaluation runs 5 speed configs, 4 accuracy datasets, 7 coverage tests (~8 min). Use to validate before committing.
- Speed improvements that break accuracy will be caught by the accuracy gate (parity < 0.85 = FAIL).
- Check `git log --oneline` to see what optimizations have already been tried.
