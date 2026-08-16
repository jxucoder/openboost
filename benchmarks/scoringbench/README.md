# OpenBoost on ScoringBench

[ScoringBench](https://github.com/jonaslandsgesell/ScoringBench) is an external
benchmark for probabilistic regression. It evaluates complete predictive
distributions with proper scoring rules and publishes accepted results at
[scoringbench.com](https://scoringbench.com/). This integration uses its dataset
loader, folds, metrics and Parquet schema without modifying the checkout.

This is the primary third-party value benchmark for OpenBoost. It answers two
different questions with two deliberately separate protocols:

1. **Official quality track**: ScoringBench's default 3,000-row cap and full
   dataset suite. These results can be proposed for its public leaderboard.
2. **Scale extension**: selected ScoringBench datasets with a larger or removed
   row cap. This measures OpenBoost's CPU/CUDA scaling but must not be presented
   as an official ScoringBench leaderboard result.

## Environment

Use a separate Linux environment because ScoringBench currently constrains
NumPy to `>=2,<2.3` and imports PyTorch for its metrics. Intel macOS is not
supported by the complete launcher: the available PyTorch wheel uses the NumPy
1.x ABI and can crash with ScoringBench's NumPy 2.x requirement. Published CPU
and CUDA measurements should come from Linux in any case. The smaller wrapper
contract test remains useful for local adapter development.

```bash
git clone https://github.com/jonaslandsgesell/ScoringBench .repos/ScoringBench
git -C .repos/ScoringBench checkout "$(cat benchmarks/scoringbench/SCORINGBENCH_COMMIT)"

uv venv .venv-scoringbench --python 3.12
uv pip install --python .venv-scoringbench/bin/python \
  -r benchmarks/scoringbench/requirements.txt
uv pip install --python .venv-scoringbench/bin/python -e .
```

`SCORINGBENCH_COMMIT` freezes the upstream protocol used for committed results.
Test newer upstream revisions separately before updating that file. The manifest
records the checked-out revision and dirty state.

On Intel macOS, the latest Numba release may not publish a compatible wheel.
The full benchmark remains unsupported there, but the wrapper contract can be
checked with the last compatible wheel instead of compiling llvmlite locally:

```bash
uv pip install --python .venv-scoringbench/bin/python 'numba==0.63.1'
uv pip install --python .venv-scoringbench/bin/python --no-deps -e .
```

For CUDA, install OpenBoost's CUDA extra using the package versions appropriate
for the benchmark machine:

```bash
uv pip install --python .venv-scoringbench/bin/python -e '.[cuda]'
```

Optional comparison models:

```bash
uv pip install --python .venv-scoringbench/bin/python xgboostlss catboost
```

## Validate the adapter

This uses one existing sklearn dataset and the complete ScoringBench metrics,
but is only an integration smoke test:

```bash
.venv-scoringbench/bin/python benchmarks/scoringbench/run.py \
  --scoringbench-dir .repos/ScoringBench \
  --models openboost_cpu,ngboost \
  --smoke \
  --n-trees 20 \
  --output-dir /tmp/openboost-scoringbench-smoke
```

The smaller wrapper contract test can also be run directly:

```bash
PYTHONPATH=.repos/ScoringBench \
  .venv-scoringbench/bin/python -m pytest \
  benchmarks/scoringbench/test_openboost_wrapper.py -q
```

The `ScoringBench` GitHub workflow runs this contract, the two-fold smoke, and
a `1027_ESL` quality sentinel on relevant pull requests, then uploads the raw
Parquet files and manifests. The PMLB/GitHub-backed sentinel avoids making the
basic quality gate depend on OpenML dataset uptime. It uses the official
five-fold, 3,000-row protocol and default 500 rounds; one dataset is still not a
quality claim. Manual `quality_shard` mode accepts an exact dataset name so the
suite can be sharded without downloading every dataset just to resolve an
index. Only a completed full suite belongs in a leaderboard submission.

## Official quality track

Run the official default: five folds, one repeat, at most 3,000 rows per
dataset. Start with OpenBoost and the existing NGBoost wrapper:

```bash
.venv-scoringbench/bin/python benchmarks/scoringbench/run.py \
  --scoringbench-dir .repos/ScoringBench \
  --models openboost_cpu,ngboost \
  --sample-size 3000 \
  --n-folds 5 \
  --n-repeats 1 \
  --output-dir benchmarks/results/scoringbench-quality
```

Use `--dataset-index N` or `--dataset-name NAME` for resumable shards. Use
`--list-datasets` to display the validated list. Do not tune OpenBoost on the
test folds. If hyperparameters are changed, apply the same declared search
budget to every comparison model.

After all shards complete, run ScoringBench's own aggregation and autoranking:

```bash
cd .repos/ScoringBench
python aggregate_datasets.py \
  --raw_dir ../../benchmarks/results/scoringbench-quality/raw \
  --out_dir ../../benchmarks/results/scoringbench-quality
python autorank_leaderboard.py --output_dir ../../benchmarks/results/scoringbench-quality
```

For an upstream submission, copy `openboost_wrapper.py` into
`scoringbench/wrappers/`, register `OpenBoostWrapper` in the upstream wrapper
exports and add a zero-argument factory to its `MODELS` dictionary. Submit the
wrapper, raw/aggregated Parquet artifacts and leaderboard JSON for independent
review.

## ScoringBench scale extension

First identify large datasets from the official list, then run the same folds
and metrics without the 3,000-row cap. CPU and CUDA are separate model names so
their results cannot be confused:

```bash
.venv-scoringbench/bin/python benchmarks/scoringbench/run.py \
  --scoringbench-dir .repos/ScoringBench \
  --models openboost_cpu,openboost_cuda,ngboost \
  --dataset-name '<EXACT_DATASET_NAME>' \
  --sample-size 0 \
  --n-folds 5 \
  --output-dir benchmarks/results/scoringbench-scale
```

Run each CUDA measurement in a fresh process. Report cold and repeated runs
separately, and include failures/OOMs. The generated `openboost_manifest.json`
records both git commits, dirty state, arguments, package versions, platform and
GPU identity. In GitHub Actions it also distinguishes the tested merge commit
from the pull request's source-head commit. Runs whose manifest says
`scoringbench_scale_extension` are not official leaderboard runs.
ScoringBench's resolved `datasets.json` is redirected into the output directory
so the exact registry travels with official artifacts without dirtying the
OpenBoost checkout.

Generated ScoringBench directories are gitignored. Publish accepted evidence in
ScoringBench's designated output/LFS repository or intentionally force-add a
frozen artifact; do not commit an arbitrary local smoke run.

The first frozen official-protocol sentinel is under
`benchmarks/evidence/scoringbench/1027_esl_20260816/`. It preserves the raw
Parquet rows and documents both favorable and unfavorable metrics. Do not
generalize that single-dataset result into a library-level claim.

## Evidence gate

OpenBoost should claim value only after all of the following are true:

- the wrapper and results are accepted upstream by ScoringBench;
- quality is reported across the full suite, not a selected winning subset;
- paired fold-level CRPS/log-score/interval-score differences include
  uncertainty intervals or the upstream statistical ranking;
- CPU and CUDA predictions pass a separate parity gate;
- a scale curve uses at least three real datasets and multiple data sizes;
- a speed claim is made only at matched predictive quality, with raw Parquet
  files and `openboost_manifest.json` published.

The benchmark is allowed to disprove the product hypothesis. If OpenBoost is
not competitive on proper scoring rules or does not accelerate at larger row
counts, the result should be published and the implementation fixed before the
README makes a performance claim.
