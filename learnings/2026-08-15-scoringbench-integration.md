# 2026-08-15: ScoringBench Integration

## Context

OpenBoost needed an existing third-party benchmark or competition to demonstrate
value. ScoringBench was selected because it evaluates full probabilistic
regression distributions with proper scoring rules and accepts upstream model
wrappers and result artifacts.

## Decision or Result

The integration has two explicitly separated protocols:

1. `official_quality`: ScoringBench's five-fold, 3,000-row protocol for an
   upstream leaderboard submission.
2. `scoringbench_scale_extension`: the same datasets/folds/metrics with a larger
   sample cap to compare OpenBoost CPU, OpenBoost CUDA, and existing baselines.

Scale-extension results must never be represented as official leaderboard
results. ScoringBench proves general probabilistic quality; it does not exercise
OpenBoost's exposure-aware API, which still needs a domain benchmark.

## Changes

- `benchmarks/scoringbench/openboost_wrapper.py`: upstream-shaped NaturalBoost
  Gaussian wrapper using ScoringBench's shared quantile-to-PMF conversion.
- `benchmarks/scoringbench/run.py`: launcher for an unmodified ScoringBench
  checkout, baseline registration, protocol labeling, and provenance manifest.
- `benchmarks/scoringbench/README.md`: environment, official track, scale track,
  upstream submission, and evidence gates.
- `.gitignore`: ignore arbitrary local ScoringBench result directories.

## Verification

- Wrapper contract: 1 passed against ScoringBench commit
  `a938a667b7839b41e9272929010573410301c0b4`.
- OpenBoost distributional regression tests: 47 passed.
- `ruff check benchmarks/scoringbench`: passed.
- Python compilation and manifest protocol classification: passed.
- Integration commit: `a4555bc` (`bench: add ScoringBench integration`).

## Failed Attempts

- The composer-swarm Cursor scout repeatedly failed with macOS Keychain error
  `SecItemCopyMatching failed -50`. Use local inspection until its CLI
  authentication is repaired; do not repeatedly retry it during one task.
- The complete ScoringBench runner is not viable on Intel macOS. ScoringBench
  requires NumPy 2.x, while the available PyTorch wheel uses the NumPy 1.x ABI.
  One run crashed and later attempts entered an uninterruptible kernel exit
  state. The launcher now refuses this platform before importing ScoringBench.
  Published CPU/CUDA runs must use Linux.

## Risks and Follow-ups

- Run the official full suite on Linux and submit the wrapper/results upstream.
- Run a separate large-sample curve on at least three real ScoringBench datasets.
- Add CPU/CUDA prediction parity before interpreting a CUDA timing result.
- Add freMTPL2 or another real exposure-aware case study after the third-party
  quality result exists.
