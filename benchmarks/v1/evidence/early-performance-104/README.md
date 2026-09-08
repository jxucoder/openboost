# Run 10: Early performance checkpoint exposes scaling limits

The frozen same-host checkpoint completes with **one passing case and five failed
cases** at clean revision `c8f7ebc57274399dbe120cd97d7933455e6559e4`. Every failure
is an incomplete measurement after a child deadline; none is relabelled a pass.
All sixteen declared case artifacts, 46 uploaded sources, 31 installed production
sources and eighteen pinned packages match. The single approved T4 allowance is
consumed. Do not rerun the command or extend its limits after seeing the results.

## Observed fit costs

All fits use sixteen numeric features, 32 bins, depth three, twenty fixed rounds,
seed seven, weights including zero-weight rows, offsets and 1% missing values.
Normal uses joint Fisher updates, producing forty terms. Validation has one fifth
as many rows as training. These are generated workloads, not real application or
external-library comparisons.

| Recipe / training rows | CPU first / warm fit (s) | GPU first / warm fit (s) | Qualification |
| --- | --- | --- | --- |
| Squared / 10,000 | 7.006 / 7.044 | 9.394 / 2.941 median | Complete; quality qualifies; CPU/GPU warm ratio **2.395** |
| Squared / 100,000 | 16.744 / 16.928 | 20.486 / 13.757, 13.748 observed | GPU child times out at 50 s before the required fourth fit completes; no ratio |
| Normal / 1,000 | 48.009 first fit only | 18.862 / 5.111 median | CPU child times out at 90 s before its second fit finishes; no ratio |
| Normal / 10,000 | No completed fit within 30 s | 21.539 / 6.858 median | CPU child times out; no ratio |
| Normal / 100,000 | No completed fit within 30 s | 49.503 first fit only | CPU and GPU children time out; no ratio |

CPU requires one first plus one warm fit. GPU requires one first plus three warm
fits; complete warm medians use all three. Individual completed fits inside a
timed-out child are observations, not completed benchmark cases. Partial artifacts
contain fit/state/counter records but no final model, predictions or quality report.
The two observed squared 100,000-row warm times therefore cannot qualify a ratio.

For the single qualifying pair, held-out half-MSE is 0.33724823350239064 on CPU
and 0.3372482702512653 on GPU. Relative metric difference is 1.0896684e-7 and
normalized prediction RMSE is 1.9468008e-7, both below the frozen 1% limits.
The fresh-process GPU first fit is slower than the CPU first fit in this case;
the reported 2.395 ratio applies specifically to warm execution.

## Where time goes

Squared 10,000-row warm training takes about 2.90 s within a 2.94 s total fit;
host binning takes about 0.012 s. Each fit makes 4,531 kernel launches and 6,272
synchronizations, with about 1.37 s in host kernel dispatch. At 100,000 rows the
same launch/synchronization counts accompany roughly 13.75 s observed warm fits
and roughly 1.42 s dispatch. Increasing row count adds substantial device/waiting
cost beyond a nearly constant host dispatch cost. No exclusive kernel attribution
is available in this checkpoint.

Normal 10,000-row warm fits make 9,417 launches and 12,982 synchronizations.
About 2.87 s of their 6.86 s fit is recorded in dispatch. The 100,000-row first
fit retains these counts, exports only 35,624 total bytes, and takes 49.50 s.
Every retained completed GPU fit reaches zero owned live bytes after run closure
and again after context closure. These counters describe the private allocation
pool, not total GPU/driver memory.

The separate Normal 100,000-row profile times out at its 60 s child limit after
one 57.953 s profiled fit. Its required warm profile is missing. Preserve this as
a failed profile case; its first-fit samples are diagnostic only:

| Profiled function | Calls | Inclusive seconds |
| --- | ---: | ---: |
| `ExecutionContext.export` | 5,553 | 28.964 |
| `DeviceOperations._launch` | 9,417 | 22.766 |
| Numba dispatcher `compile` | 24 | 15.694 |
| `ExecutionContext.release` | 6,533 | 1.189 |
| `ExecutionContext.synchronize` | 7,429 | 1.166 |
| `ExecutionContext.upload` | 50 | 0.092 |

These inclusive costs overlap. Blocking exports wait for preceding kernels, so
28.964 s is not transfer bandwidth cost. Compiler calls include dispatch/cache
work and are nested in launch cost. Neither the profile nor the first/warm
difference measures pure compilation time. Source inspection identifies serial
row scans in validation/reduction kernels and a full routed-row scan per histogram
cell. Their exact contribution is a hypothesis requiring targeted measurement.

## Supported inference

Saved models run through the existing CPU prediction path. The fully saved GPU
models have warm median single-row / validation-batch prediction times of:

| Model | Single row (ms) | Validation batch (ms) | Batch rows |
| --- | ---: | ---: | ---: |
| Squared / 10,000 train | 3.482 | 35.574 | 2,000 |
| Normal / 1,000 train | 6.904 | 16.468 | 200 |
| Normal / 10,000 train | 6.795 | 70.828 | 2,000 |

These are CPU inference measurements of GPU-trained models. They do not establish
GPU inference, competitive inference performance or startup/import cost. The raw
artifacts also retain model loading, all prediction repetitions and CPU-model
prediction measurements.

## Environment, provenance and reproduction limits

The request uses one T4, two CPUs, 8192 MiB, one container, 900 function seconds,
600 test seconds and zero retries. Guest-visible CPU count is eighteen; it is not
the requested CPU entitlement. GPU reports Tesla T4 / 15360 MiB / driver 580.95.05.
Runtime/driver API versions are 12090/13000. Python is 3.12.1 on Linux 4.19.0,
gVisor, x86_64, glibc 2.35. NumPy is 2.3.5; the manifest retains every pinned
version and the isolated NumPy/core CPU-environment build record. Each child uses
one BLAS/OpenMP thread and its own CuPy/driver cache paths.

Dispatch runs from 2026-09-08 07:06:15.372134 UTC to 07:15:22.469733 UTC,
547.097599 s including image construction/setup. Worker elapsed time is
463.174024267 s; JUnit records 462.376 s. Child deadlines total at most 565 s.
First-fit timers include context setup and triggered JIT; total fit includes
preparation, training, export and cleanup, while dataset construction, inference,
JSON serialization and imports outside the timer remain separately recorded or
included only in child wall time. Warm fits use fresh contexts in the same child
process, with compiled kernels reusable. This is not a universally cold driver.

The Linux judge independently recomputes held-out scores and verifies identical
CPU/GPU input identities for the qualifying pair. On the offline macOS x86_64
audit host, the same NumPy version and generator do **not** reproduce the remote
Problem identities. The exact cause is unisolated; using double-precision sine
alone does not recover the identity. All five fully saved CPU/GPU models still
replay bit-exactly on locally generated validation features. Local recomputed
metric differences are at most 4.74e-9 in these records, but those computations
use different input identities and cannot replace the original remote audit.

Consequently, `analyze.py` verifies frozen revision/source bindings, every raw hash,
case identities, completed model/state/cleanup records, local prediction replay,
and eligibility/ratios from the retained same-host scores. It reports local input
regeneration separately; it does not claim a portable exact rerun of the frozen
metric judge. The next packet must retain exact generated inputs. Changing the
dataset or loosening a gate cannot retroactively complete this run.

## Artifacts and next decision

- [manifest.json](manifest.json), [junit.xml](junit.xml), [pytest.log](pytest.log),
  [verdict.json](verdict.json): literal failed execution and provenance.
- `normal/checkpoint/`: sixteen declared child/judgment/profile JSON artifacts,
  including all partial results.
- [analysis.json](analysis.json), [analyze.py](analyze.py): derived measurements
  and offline verification, with explicit regeneration limitations.
- [archive-index.json](archive-index.json): hashes of all other archive files.

Offline audit command, requiring the unchanged production/benchmark implementation:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python benchmarks/v1/evidence/early-performance-104/analyze.py --check
```

[Sprint 104](../../../../v1-sprints/104-early-performance-checkpoint.md) records the
retrospective. The next proposed bounded slice is parallel boolean/domain
validation, paired with exact input snapshots and feasible repetition budgets.
Preserve all validation/ownership and comparison semantics; numerical reductions
need their own correctness design. This result does not pass formal E4, establish
external-library speed, close all required CUDA recipes or show adoption benefit.
