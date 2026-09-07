# Sprint 070: Real A6 comparator resource probes

Clean source `02f3d41`. Three frozen fold-zero configuration-00 jobs from the
updated 400-job A6 CPU plan pass. The already approved 1494662-byte Parkinsons
train/validation packet and 34 allowlisted public files were uploaded. No test
features or test labels were uploaded or scored. Exact CLI, source revision,
image identity, packet/plan hashes, environment and launch commands are retained.

## Results

| Comparator | Fit worker seconds | Fresh replay seconds | Recorded rounds | Selected rounds | Peak guest RSS bytes |
| --- | ---: | ---: | --- | --- | ---: |
| XGBoost | 4.667475343 | 3.139234259 | 59 | 9 | 325197824 |
| LightGBM | 3.856247621 | 3.048936888 | 71 / 76 | 21 / 26 | 261861376 |
| CatBoost | 1.836083715 | 1.428013208 | 56 | 6 | 429481984 |

All have the frozen 300-round maximum and patience 50. LightGBM has independent
target models and separate native stopping histories. XGBoost uses shared vector
trees; CatBoost uses MultiRMSE. Histories are retained without reducing the budget.
Every model replays exactly in a fresh process, including validation row IDs.
The saved training-only target scale matches the earlier verified OpenBoost fold
freeze exactly. No comparative quality metric or selected-test release is inferred.

Each fit runs with UID/GID 65534, no-new-privileges, an 8-GiB address ceiling,
one BLAS thread and 1800 seconds. Modal requests two CPUs, 8192 MiB and a
1900-second function limit per job, with zero retries and sequential dispatch.
Each fresh replay has 60 seconds. All worker exit codes are zero. The processes
report XGBoost 3.4.1, LightGBM 4.7.0, CatBoost 1.2.10; full installed dependency
versions are in resources.json. Platform/Python/NumPy are in manifest.json.
Host CPU model, physical RAM and cgroup enforcement are not captured. Guest RSS
is distinct from the address limit and requested container memory capacity.

Timings include each worker's imports, preprocessing, fit and serialization;
replay includes imports and model loading. Upload/image/startup/coordinator cost
is excluded. The three jobs are not a repeated or matched-quality speed study.

## Verification and reproduction

All 34 source hashes verify against the clean revision. All 30 outer raw artifact
hashes and fifteen inner worker artifact hashes verify. Each execution record
matches its manifest copy. All replay arrays compare exactly. The resource-plan
and original input-packet hashes verify. The original input packet is reproducible
through the frozen exporter; returned validation feature packets are retained.

```bash
uv run --no-sync python -m benchmarks.v1.worker_data A6 /tmp/a6-packets
uv run --no-sync python -m benchmarks.v1.a6_resource_preflight /tmp/a6-comparators /tmp/a6-packets --comparators
```

The preparation commit passed 1120 CPU tests, one Linux-only skip; 28 focused
checks, lint and docs pass. No failed probe or retry occurred in this run.

## Reflection and next step

The first three real comparator jobs satisfy the bounded execution/replay check.
The other 237 comparator jobs, deeper/1000-round cases and complete selection
remain unqualified. These artifacts do not establish the complete 400-job search,
R/C/A/E coverage, authoring cost, quality, adoption or GPU gates.

The comparator workers finish in seconds, while the earlier current OpenBoost
shared probe took about 757 seconds. Different hosts and algorithm/selection
semantics prevent interpreting this as a precise speed ratio, but it is a material
practical CPU runtime warning. The foundation must be judged on verified algorithm
changes and usable execution; a 23.6% internal improvement is not competitive-cost
evidence. Keep this warning visible in the next retrospective. Before expanding
expensive OpenBoost searches, qualify deeper/1000-round resource cases and retain
failures under the existing budgets. Author-accounting and full coverage work
remain necessary; do not automatically launch the full matrix or another unbounded
optimization effort from these three successes.
