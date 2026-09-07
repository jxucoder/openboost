# Sprint 067 incremental execution evidence

[counts.json](counts.json) records clean revision `e99e89c` and exact synthetic
full replay for squared and Normal at 4/8/16/32 rounds. Counts are exactly 2*K*T:
one evaluation per new tree for training and validation. These are operation counts,
not time or peak memory. Historical quadratic evidence remains in runtime-audit-063.

[Installed checks](installed/manifest.json) rebuild the core and four development
extension wheels, exercise direct and M=1/8/32 scheduling, custom stopping and
ordered updates, then remove training plugins and exactly replay ten saved models
in fresh inference. This is internal conformance, not independent-author evidence.
All artifact and source hashes were independently verified against the revision.

Commands (with `UV_CACHE_DIR=/tmp/openboost-research-uv-cache`):

- `uv run --no-sync python -m benchmarks.v1.runtime_cost_audit /tmp/openboost-counts-067.json`
- `uv run --no-sync python examples/v1_extensions/verify.py /tmp/openboost-v1-sprint067`

## Frozen practical sweep

[Practical manifest](practical/manifest.json) records clean revision `e99e89c`,
all eight uninstrumented fits and both instrumented profiles passing under the
original frozen inputs, settings and caps. Actual-image resource probes and
rejection/backtracking fixtures pass. Each fit verifies exact full replay and
persisted inference; raw metrics and artifact/source hashes were independently
checked. Full-model verification is outside timed fit/prediction/export intervals.

| Recipe | Training rows | Rounds | Fit seconds | Guest peak MiB | Trace arrays MiB |
| --- | ---: | ---: | ---: | ---: | ---: |

| squared | 8192 | 4 | 0.2188 | 83.38 | 0.562 |
| normal | 8192 | 4 | 0.4331 | 85.09 | 2.125 |
| squared | 2048 | 32 | 1.5649 | 84.98 | 1.016 |
| normal | 2048 | 32 | 3.1307 | 87.43 | 4.031 |
| squared | 8192 | 32 | 1.8208 | 88.54 | 4.062 |
| normal | 8192 | 32 | 3.6598 | 101.06 | 16.125 |
| squared | 8192 | 128 | 9.2532 | 101.38 | 16.062 |
| normal | 8192 | 128 | 18.5037 | 155.03 | 64.125 |

No completed 128-round baseline exists; do not infer a speedup there. Cross-run
OpenBLAS architecture changed from Haswell to SkylakeX. Squared predictions/models
are byte-exact across runs; Normal raw predictions differ by at most 8.9e-16.
Cross-host timing ratios are not attributed solely to the optimization.

Both profiles complete: squared 128 rounds uses 256 tree predictions and three
encoding transforms in 13.241 instrumented seconds; Normal uses 512 predictions
and three transforms in 26.415 seconds. Tree prediction cumulative time is 0.039
and 0.072 seconds respectively, below 0.3% of each fit. Tree construction is now
10.334/20.435 seconds (about 78%/77%); cumulative times overlap. These profiles
support removal of repeated replay/encoding, not linear complexity of all work.
Full trace arrays remain unchanged in policy; their logical bytes are not RSS.

The same-container paired amendment and results below separate the CPU architecture
confound from the algorithm comparison. No formal quality or cost gate is claimed.

## Same-container paired comparison

The [paired manifest](paired/manifest.json) records revision `da0fede`, the retained
baseline wheel hash, twelve ordered fits and unchanged per-worker caps. This
separate amendment runs only six completed baseline cases; it does not extend the
original eight-fit budget silently. Both variants use the same container and
SkylakeX OpenBLAS pool, two threads, dependencies, input and algorithm settings.
Recorded loaded runtime/recipes/tree source hashes verify each wheel's actual code.

All six pairs have **exact raw predictions and byte-identical model JSON**.
[Comparison](paired/comparison.json) records both fit and end-to-end ratios.
The earlier cross-host Normal difference is absent in this controlled comparison;
no tolerance change was needed.

| Case | Baseline fit s | Candidate fit s | Observed fit ratio | End-to-end ratio |
| --- | ---: | ---: | ---: | ---: |
| squared-8192-4 | 0.3148 | 0.2208 | 1.43 | 1.37 |
| normal-8192-4 | 0.5878 | 0.4281 | 1.37 | 1.34 |
| squared-2048-32 | 2.9735 | 1.5435 | 1.93 | 1.90 |
| normal-2048-32 | 5.8634 | 3.1983 | 1.83 | 1.81 |
| squared-8192-32 | 8.1357 | 1.7834 | 4.56 | 4.37 |
| normal-8192-32 | 11.9292 | 3.6436 | 3.27 | 3.17 |

Each row is one ordered pair, baseline first, with no randomized order or repeats.
Ratios describe these observations, not a stable speed guarantee. End-to-end sums
record setup, complete fit including binning/validation, prediction and export;
outer subprocess wall additionally includes imports/loading/verification. Model
quality is identical within each pair. No XGBoost/LightGBM/CatBoost comparison or
formal E4 acceptance follows. All raw source/artifact hashes are independently
verified; generated profile text retains its original trailing blank line.
