# OpenBoost Agent Guide

This is the canonical repository guidance for coding agents and automated
contributors. Tool-specific instruction files should point here instead of
duplicating policy.

## Mission

OpenBoost is a readable Python gradient-boosting research platform. Its current
product focus is **calibration-first distributional boosting for tabular risk**:
NaturalBoost, proper scoring, calibration, exposure-aware targets, custom
distributions, and verified CPU/CUDA execution.

Do not position the repository as a drop-in replacement for XGBoost, LightGBM,
or CatBoost. Standard GBDT, GAM, DART, linear leaves, Ray, multi-GPU, and
train-many are supporting or experimental capabilities unless a committed,
reproducible artifact proves otherwise.

## Start Here

Before a non-trivial change:

1. Read this file and the relevant entries in `learnings/`.
2. Check `git status --short --branch`; preserve unrelated user changes.
3. Read the implementation, its tests, and the public documentation together.
4. Write a short plan for work spanning three or more meaningful steps.
5. Identify the smallest test that can fail before editing.

Do not trust phase comments, docstrings, README claims, or green CI as proof by
themselves. Verify the actual call path and the tests that exercise it.

## Current Priority Order

1. Silent correctness and persistence bugs.
2. Deterministic CPU behavior and reference parity.
3. One verified single-GPU NaturalBoost path, including end-to-end quality.
4. Third-party evidence through ScoringBench and real domain case studies.
5. Stable packaging, versioned persistence, and a smaller public API.
6. New features only after the above gates are satisfied.

Treat Ray, multi-GPU, out-of-core training, GOSS speedups, and fused train-many
as experimental. Do not expand or market them until exact correctness and
scaling artifacts exist. The repository audit in
`learnings/2026-08-15-repository-audit.md` records the current evidence gaps.

## Architecture

```text
Models (`src/openboost/_models/`)
    -> tree core (`src/openboost/_core/`)
        -> CPU/CUDA backends (`src/openboost/_backends/`)
    -> distributions (`src/openboost/_distributions.py`)
    -> validation and persistence
```

- `BinnedArray` is feature-major: `(n_features, n_samples)`.
- Bin 255 is reserved for missing values; use at most 254 regular bins.
- The backend is process-global, not thread-local. Use `backend_context` for a
  scoped switch and do not run mixed-backend fits concurrently in one process.
- NaturalBoost fits one tree per distribution parameter per round.
- CUDA eligibility is narrower than the public model surface. A fallback must
  be visible, tested, and represented honestly in benchmark provenance.

## Correctness Rules

- A serialization change requires prediction round trips for numeric,
  categorical, missing-value, and specialized leaf/tree state that it touches.
- A CUDA change requires CPU/CUDA parity for gradients, splits, leaves,
  predictions, and final task metrics—not just matching array shapes.
- Never silently ignore `sample_weight`, exposure, callbacks, evaluation sets,
  constraints, or sampling parameters. Support them or reject them explicitly.
- Randomized behavior must be driven by the model's declared seed; do not use
  unscoped global `numpy.random` state.
- Distributed child histograms must be derived from routed samples. Scaling a
  parent histogram is not an exact substitute.
- Public examples are tests of product behavior. If an example cannot run, fix
  it or label the feature experimental before documenting it.

## Evidence and Benchmark Rules

- Every performance or quality claim must link to a committed raw artifact.
- Record git SHA and dirty state, dataset/version/hash, split seed, package
  versions, OS, CPU/RAM/thread count, GPU/driver/CUDA, and exact CLI arguments.
- Compare end-to-end fit and prediction, including distribution gradients,
  transfers, compilation policy, and fallbacks. Kernel microbenchmarks cannot
  support an end-to-end product claim.
- Use repeated folds/seeds and publish failures. Compare at matched predictive
  quality; do not declare a speed win when CRPS/NLL/calibration regresses.
- Keep official ScoringBench results separate from OpenBoost's large-sample
  extension. See `benchmarks/scoringbench/README.md`.
- Synthetic experiments generate hypotheses. Real third-party datasets and
  upstream-accepted results generate evidence.

## Commands

Use `uv`; do not mutate the project environment with ad-hoc `pip` or Conda
commands.

```bash
# Install
uv sync --extra dev
uv sync --extra cuda

# Focused test while iterating
OPENBOOST_BACKEND=cpu uv run pytest tests/test_file.py -n 0 -q

# CPU regression suite
OPENBOOST_BACKEND=cpu uv run pytest tests/ -m "not gpu and not benchmark" --tb=short

# Lint production code and changed support files
uv run ruff check src/openboost/ path/to/changed_file.py

# Documentation and packaging
uv run mkdocs build
uv build
```

Run CUDA tests only on real CUDA hardware. A skipped GPU job is not a passing
GPU validation. ScoringBench has a separate Linux environment documented under
`benchmarks/scoringbench/`.

## Working and Commit Discipline

- Keep changes small and cohesive. Prefer root-cause fixes over compatibility
  shims that conceal invalid state.
- Commit after each independently verified slice: test/benchmark harness,
  correctness fix, documentation/learning update, or infrastructure change.
- Do not bundle unrelated cleanup into a fix. Do not amend or rewrite existing
  commits unless the user explicitly asks.
- Do not push, publish, create a release, or update an external leaderboard
  unless the user asks for that external action.
- Before every commit: inspect the staged diff, run the narrowest meaningful
  tests, and include the verification in the relevant learning entry.

## Learning Log

`learnings/` is the durable project memory for decisions, failed attempts,
experiments, and non-obvious operational facts.

- Add or update an entry for every non-trivial change.
- Use `learnings/TEMPLATE.md`.
- Record evidence and falsified hypotheses, not a diary of shell commands.
- Link files and commits. State what was not verified.
- Never include credentials, tokens, private URLs, or user-specific secrets.
- Release notes describe user-facing changes; learning entries explain why the
  implementation and evidence changed.

## Definition of Done

A change is done only when:

1. The intended behavior is covered by a focused test or reproducible artifact.
2. Relevant regression tests and lint pass.
3. Documentation and capability claims match the implemented boundary.
4. A learning entry captures important decisions, failures, and follow-ups.
5. The change is committed as a cohesive unit and the remaining work is stated.
