# Sprint 070: One paired real A6 fit

Current clean revision: `73755b1`; original public CPU baseline: `17ab9de`.
Only `src/openboost/ops.py` differs between these public CPU snapshots. Both fits
use the current frozen worker and the same Parkinsons fold-zero configuration 00:
shared topology, 300-round budget, patience 50, depth 4, 255 bins, learning rate
0.03, zero regularization and summary retention. No test data was uploaded or scored.

## Results

| Variant | Fit worker wall seconds | Fresh replay wall seconds | Fit plus replay seconds | Completed rounds | Peak guest RSS bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original baseline | 992.047390528 | 0.319307054 | 992.366697582 | 59 | 107200512 |
| Current | 757.479782603 | 0.269394496 | 757.749177099 | 59 | 106422272 |

The observed fit time decreases by 23.6448% in this one fixed-order pair. Both
variants stop by patience and select exactly the same best-validation model.
Model bytes, saved predictions, training/stopping records and fresh replay archives
are byte-identical across variants and with the earlier original shared real probe.
Fresh replay arrays, including row IDs, independently compare exactly.

This is one same-container observation, baseline first, not a repeated timing
estimate or a general performance claim. Order/cache effects are not balanced.
The baseline is imported from `/baseline/openboost`; current is installed under
Python's site-packages. Both loaded paths are recorded in resources.json. The
source-vs-installed distinction remains part of the experiment, though only ops.py
differs in public production Python content. Fit timing includes worker startup,
preprocessing, fitting and artifact serialization. Replay includes a fresh Python
process and saved-model inference. Image build/upload and coordinator overhead are
excluded from these timings; these are not total cloud cost figures.

## Execution and provenance

One Modal function runs the two variants sequentially with identical dependencies:
Python 3.12.10, NumPy 2.3.5, Linux gVisor x86_64 with glibc 2.36. The image requests
two CPUs and 8192 MiB, has a 3800-second limit and zero application retries. Each
fit uses the existing minimal environment, UID/GID 65534, no-new-privileges, one
BLAS thread, an 8-GiB address ceiling and 1800-second timeout. Each replay has a
60-second timeout. Both exit successfully. CPU model/physical host RAM and host
cgroup enforcement are not captured; requested capacity is not measured capacity.
Peak guest RSS is not address space or the requested container memory limit.

The upload contains the same approved 1494662-byte train/validation-only packet,
32 current public source/packaging/support files, and 20 frozen baseline public
Python files. No Git history or evaluator/test-label material is uploaded. The
packet has 3487 training rows, 1151 validation rows, 38 features and two UPDRS
targets; its original hash and frozen preprocessing metadata are in manifest.json.
Source paths, full Git revisions, dirty state, image identity, package versions,
CLI arguments and exact fit launch commands are retained there and in execution
records. The original worker packet is reproducible through the verified exporter;
it is not duplicated here. Returned validation feature packets are retained.

## Verification

All 32 current source hashes verify against `73755b1`; all 20 baseline hashes
verify against `17ab9de`; all 40 raw artifact hashes verify. The original packet
and frozen resource-plan hashes verify. Actual package paths confirm the intended
baseline/current import roots. All four paired model/prediction/training/replay
artifacts compare byte-for-byte; selected model identity remains
`b7759ae2782eecfd9cceb6220260a973b6081d051055bc7b1e9e1a9eb36e19d5`.
Both training records retain identical train-only target scales and stop state.

```bash
uv run --no-sync python -m benchmarks.v1.worker_data A6 /tmp/a6-packets
uv run --no-sync python -m benchmarks.v1.a6_resource_preflight /tmp/a6-pair /tmp/a6-packets --paired
```

The harness commit passed 23 focused contracts, 1095 CPU tests with one Linux-only
skip, lint and documentation build. The real pair adds actual Linux fit/replay
evidence for this configuration; it does not replace the skipped unrelated test.

## Reflection and next step

The two measured scoring changes preserve the real selected result and reduce
observed full-fit time. End the optimization detour here. A 757-second fit is still
substantial for one modest configuration, so do not extrapolate the saving into
an acceptable full-search cost. The other 158 OpenBoost jobs, deeper/1000-round
configurations, comparative quality, authoring, adoption and CUDA gates remain open.

Return to Sprint 070's evaluator-owned complete matrix and comparator/resource
coverage, alongside Sprint 069's executable accounting/isolation packet. Any later
resource preflight must keep the frozen budgets and retain failures; no posthoc
search shrink or automatic full-matrix launch follows this successful pair.
