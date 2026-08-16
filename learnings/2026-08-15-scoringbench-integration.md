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
- `benchmark_outcome.json`: an exact dataset/model/fold completeness audit that
  makes upstream-captured failures and non-finite distributional metrics
  machine-readable and changes the launcher exit status to failure.
- Frozen-registry sharding: `--dataset-registry`, `--shard-index`, and
  `--shard-count` split one ordered registry across workers without rebuilding
  a potentially changing OpenML suite in each job; source and copied-registry
  hashes are recorded in schema-v2 manifests.
- Strong-baseline mode adds the ScoringBench native XGBoost quantile, Gaussian
  XGBoostLSS, and CatBoost MultiQuantile wrappers with frozen package versions
  and their registered model-specific budgets. This makes NGBoost a reference,
  not the acceptance bar.
- Development mode exposes OpenBoost tree count, learning rate, depth, L2 leaf
  regularization, and minimum child Hessian while forcing the manifest protocol
  label to `development_tuning`. It preserves ScoringBench folds and metrics but
  is intentionally ineligible for held-out or leaderboard evidence.

## Verification

- Wrapper contract: 1 passed against ScoringBench commit
  `a938a667b7839b41e9272929010573410301c0b4`.
- Fresh isolated-environment contract after adding xdist: 1 passed on Intel
  macOS with Numba 0.63.1; this validates the adapter only, not benchmark scores.
- OpenBoost distributional regression tests: 47 passed.
- ScoringBench provenance/outcome/sharding tests: 8 passed after adding frozen
  registry loading, exact-completion and mixed missing/error/non-finite cases,
  and proof that strided shards cover each entry exactly once.
- Strong-baseline parser/provenance suite: 9 passed. The frozen Linux target
  dependency contract resolved 87 packages including NumPy 2.2.6, Pandas
  2.2.3, Torch 2.9.1, XGBoost 3.3.0, XGBoostLSS 0.6.1, and CatBoost 1.2.10.
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
- Linux ScoringBench run #6 completed the `1027_ESL` official-quality sentinel:
  10 fold/model rows, clean provenance, and artifact verification all passed.
  Artifact `9256929555` has digest
  `sha256:0c5176c4a28cd44444f6a086643a01f636b0dcb5e165a4a30d3d992f3e96da97`.
- The frozen evidence under
  `benchmarks/evidence/scoringbench/1027_esl_20260816/` preserves the manifest,
  resolved dataset registry, both raw Parquet files, and a descriptive summary.
  OpenBoost's mean CRPS/RMSE/90% interval score and time were better on this
  shard, but mean log score was worse and CRPS won only 2/5 folds.
- Clean strong-baseline run `31925701435` completed all 25 expected rows for
  `1027_ESL` at source commit `cea891a` and pinned ScoringBench commit
  `a938a667`. Artifact `9257853524` has digest
  `sha256:4044cc803958036d16c55aefed98c3142486e7ddba4bdfca61f364d5e7310765`.
  The source checkout had no porcelain changes, and every frozen input/result
  file is checksummed in
  `benchmarks/evidence/scoringbench/1027_esl_strong_20260816/summary.json`.
- `1028_SWD` clean baseline run `31926255124` completed 25/25 rows in artifact
  `9258026048` (digest `sha256:495ac3c8c02c0846423131c4636dfde05f0da872b637bca2be9493b3c02aa8b4`).
  Development run `31926664340` completed 5/5 rows with the expected
  `development_tuning` label in artifact `9258115955` (digest
  `sha256:b3b77cbf691bef574b0a27c897b8a92a22e90a4c158f0f7e3ae556400d60e599`).
  Both source checkouts were clean; every copied artifact file is checksummed
  under `benchmarks/evidence/scoringbench/development/1028_swd_lr_sweep_20260816/`.
- On that one diagnostic shard, OpenBoost ranked first on mean CRPS, 90%
  interval score, absolute 90% coverage error, and PIT KS. Against native
  XGBoost quantile it reduced those metrics by 7.7%, 38.8%, 82.2%, and 59.5%
  respectively and reduced RMSE by 11.6%. Against Gaussian XGBoostLSS it had
  1.2% lower CRPS and 51.3% lower coverage error, but 1.4% worse RMSE and about
  9.1 times its fit-only per-fold training time.
- Those results define a useful hypothesis, not a win: the shard has one small
  dataset, one seed, correlated folds, unequal model-specific budgets, CPU
  only, and no confidence interval. Timing also varied materially between two
  otherwise equivalent Actions runs, so it cannot support a speed claim.
- A post-run metric audit found that the current ScoringBench reconstructed
  log score clamps targets outside finite quantile support to the boundary-bin
  density. CRLS is integrated over each model's own support, and the upstream
  implementation explicitly warns that values are not comparable across
  different bin grids. Both metrics remain in the raw artifact but are excluded
  from cross-model conclusions. Use a separately audited analytic Gaussian NLL
  for parametric-only density comparison; do not optimize OpenBoost against the
  current quantile log-score artifact.
- Untouched development dataset `1028_SWD` showed the opposite CRPS ranking
  from `1027_ESL`: OpenBoost had the best interval score, coverage error, and
  PIT KS, but mean CRPS was 4.9% worse than native XGBoost quantile, 1.7% worse
  than XGBoostLSS, and 2.0% worse than NGBoost. It lost CRPS to native XGBoost
  on all five folds. This disproves a general quality-win claim and identifies
  a reproducible sharpness/calibration trade-off.
- A development-only `0.03 × 500` OpenBoost run narrowed mean sharpness by 5.4%
  relative to `0.01 × 500`, but worsened CRPS by 0.44%, interval score by 4.74%,
  PIT KS by 9.82%, and absolute 90% coverage error by 3.23%. It improved paired
  CRPS in only two of five folds. Reject the larger fixed learning rate; the
  next experiment must separate mean accuracy from post-fit scale calibration
  or test a different scale objective.
- Gaussian CRPS training is not a new algorithmic claim: NGBoost already
  publishes a Normal CRPS score and generalized natural-gradient metric. A
  direct local prototype of that metric was unstable at larger learning rates
  on heteroscedastic synthetic data (mean predicted scale exploded), so it was
  rejected as OpenBoost's implementation path rather than copied blindly.
- The exact Gaussian CRPS Hessian is indefinite in the tails and cannot be fed
  to OpenBoost's positive-Hessian tree solver. The implemented explicit
  `training_objective='crps'` instead uses the strictly positive expected CRPS
  curvature under the current Normal prediction. This keeps the default NLL
  path unchanged, keeps training objective independent from `eval_metric`, and
  fails early for non-Normal distributions.
- In a local heteroscedastic synthetic diagnostic, expected-curvature CRPS
  training reduced held-out CRPS relative to NLL training at each tested fixed
  learning rate (`0.003`, `0.01`, `0.03`, and `0.1`) for 300 depth-3 rounds.
  This is a development hypothesis only, not benchmark evidence. The core
  mathematical/API slice passed 191 tests (2 skipped), including finite
  differences, default-NLL identity, objective logging, sklearn cloning, and
  persistence. It must still win on the frozen `1028_SWD` development folds
  before being promoted into the ScoringBench wrapper experiment.
- The benchmark launcher and manual Actions workflow now record and forward
  `training_objective`; the default remains `nll`, while CRPS candidates must
  carry the `development_tuning` protocol label. The provenance suite passed
  10 tests and both NLL/CRPS wrapper contracts passed against the pinned
  ScoringBench checkout.
- Integration commit: `a4555bc` (`bench: add ScoringBench integration`).

## Failed Attempts

- A first local dependency resolution appeared to make XGBoostLSS 0.6.1 select
  Torch 2.2.2 beside NumPy 2.2.6. PyPI metadata disproved the suspected hard
  pin: XGBoostLSS declares `torch>=2.1,<2.10`. The old selection came from
  resolving for Intel macOS, where 2.2.2 is the final available Torch wheel;
  benchmark dependencies must be resolved for the Linux target platform.
- Strong-baseline run `31925230473` completed all 25 expected rows, but its
  OpenBoost checkout was dirty after training. CatBoost writes `catboost_info/`
  in the process working directory unless configured otherwise. The factory
  now sets `allow_writing_files=False`, manifests include porcelain change
  paths, and the artifact gate rejects a dirty OpenBoost checkout.

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
- ScoringBench's outer runner catches dataset exceptions and its fold runner
  converts model exceptions into rows; neither condition necessarily produces
  a failing process. The OpenBoost launcher now audits the returned records
  after the entire shard finishes, preserves all omissions/errors, and only
  then exits non-zero for an incomplete outcome.

## Risks and Follow-ups

- Run the official full suite on Linux and submit the wrapper/results upstream.
- Treat `1027_ESL` as consumed diagnostic evidence. Do not tune on it and then
  relabel the result as held out. Use separate development datasets to test
  Normal scale/log-score hypotheses, preserve CRPS/calibration guardrails, and
  make the final decision on untouched datasets or the complete suite.
- The acceptance bar is stronger than NGBoost parity: OpenBoost should rank
  first or statistically tied on the primary proper scores, beat the strongest
  XGBoost-family baseline on a majority of paired datasets, and avoid material
  regressions in interval score, calibration, RMSE, failure rate, or resource
  use. CRPS is the current cross-model primary metric. Density scoring becomes
  a guardrail only after common-support handling is validated; report
  per-dataset paired effects and uncertainty, not only macro means.
- Run a separate large-sample curve on at least three real ScoringBench datasets.
- Add CPU/CUDA prediction parity before interpreting a CUDA timing result.
- The first artifact identified the PR merge commit but not the source-head SHA.
  The manifest now records both; confirm the mapping in the next CI artifact.
- Add freMTPL2 or another real exposure-aware case study after the third-party
  quality result exists.
