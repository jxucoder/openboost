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
- `.github/workflows/scoringbench.yml`: pinned Linux contract/smoke validation,
  artifact upload, and a manually dispatched official-quality shard.

## Verification

- Wrapper contract: 1 passed against ScoringBench commit
  `a938a667b7839b41e9272929010573410301c0b4`.
- Fresh isolated-environment contract after adding xdist: 1 passed on Intel
  macOS with Numba 0.63.1; this validates the adapter only, not benchmark scores.
- OpenBoost distributional regression tests: 47 passed.
- `ruff check benchmarks/scoringbench`: passed.
- Python compilation and manifest protocol classification: passed.
- GitHub workflow YAML parsed locally; the pinned wrapper contract passed before
  the workflow was added, then the complete Linux smoke passed in run #1.
- Linux ScoringBench run #1 passed the contract, two-fold OpenBoost/NGBoost
  smoke, artifact verification, and upload. Artifact `9256565927` contains the
  manifest and both raw Parquet files with 4 result rows; its digest is
  `sha256:3c98f640af58291f7cb648ac38bfa04ff0a44bd198698fb47a2b4f22dfb98862`.
  This proves the integration path only, not comparative model value.
- CI/source provenance, artifact-working-directory, and named-shard selection
  tests: 4 passed.
- Linux ScoringBench run #4 confirmed `source_sha`, tested PR merge SHA, clean
  checkouts, pinned upstream SHA, and 4 smoke rows in artifact `9256692529`.
- Integration commit: `a4555bc` (`bench: add ScoringBench integration`).

## Failed Attempts

- The composer-swarm Cursor scout repeatedly failed with macOS Keychain error
  `SecItemCopyMatching failed -50`. Use local inspection until its CLI
  authentication is repaired; do not repeatedly retry it during one task.
- The complete ScoringBench runner is not viable on Intel macOS. ScoringBench
  requires NumPy 2.x, while the available PyTorch wheel uses the NumPy 1.x ABI.
  One run crashed and later attempts entered an uninterruptible kernel exit
  state. The launcher now refuses this platform before importing ScoringBench.
- A fresh isolated environment could not collect the wrapper test because the
  repository-wide pytest configuration enables xdist while the benchmark
  requirements omitted `pytest-xdist`. The isolated requirements now include
  it.
- Installing unconstrained `numba>=0.60` on Intel macOS selected Numba 0.67,
  for which no compatible wheel was available; llvmlite then tried to build
  against LLVM 20 although that release requires LLVM 22. Installing the last
  available Intel wheel (`numba==0.63.1`) and the editable project with
  `--no-deps` is sufficient for the wrapper-only contract. This workaround is
  not a supported full benchmark environment. Published CPU/CUDA runs must use
  Linux.
- Building ScoringBench's official dataset registry writes `datasets.json` to
  `Path.cwd()`. A dataset-list probe therefore polluted the OpenBoost root and
  would make a later manifest report a dirty checkout. The launcher now builds
  and validates that registry from inside the output directory and restores the
  original working directory even after an exception.
- ScoringBench validates datasets by loading them. Selecting a shard by index
  therefore validates the entire registry before the index exists. Exact-name
  shards now select first and validate only the requested datasets; index and
  list modes retain their validated-list semantics.
- The first Abalone sentinel never reached model fitting: all three OpenML suite
  requests returned server errors and dataset 183 then exhausted retries with
  HTTP 504. ScoringBench caught the dataset exception and returned an empty
  result while exiting successfully; OpenBoost's `result_rows == 10` artifact
  gate correctly failed the job. The PR sentinel now uses ScoringBench's
  `1027_ESL` PMLB/GitHub source so this small gate does not depend on the OpenML
  data endpoint. Full-suite runs still must report OpenML failures rather than
  silently treating them as model results.

## Risks and Follow-ups

- Run the official full suite on Linux and submit the wrapper/results upstream.
- Use the `1027_ESL` sentinel only as a reproducibility/integration gate; inspect
  its 5-fold artifact before deciding whether OpenBoost merits broader shards.
- Run a separate large-sample curve on at least three real ScoringBench datasets.
- Add CPU/CUDA prediction parity before interpreting a CUDA timing result.
- The first artifact identified the PR merge commit but not the source-head SHA.
  The manifest now records both; confirm the mapping in the next CI artifact.
- Add freMTPL2 or another real exposure-aware case study after the third-party
  quality result exists.
