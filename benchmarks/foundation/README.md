# Foundation GPU evidence

This entrypoint is isolated from `tests/modal_gpu_tests.py`: the legacy app
registers source-mounted jobs and broad dependencies. Loading it would defeat
the installed-wheel boundary and build unrelated images. Existing jobs remain
available; this app uploads only its allowlisted bundle with automatic source
inclusion disabled.

From a clean committed checkout:

```bash
uv run python -m benchmarks.foundation.prepare
uv run modal run benchmarks/foundation/modal_app.py::foundation_smoke
```

The bundle is generated in ignored `build/foundation/`. Stale/dirty source,
modified bundle files, failing/missing/skipped/duplicate required tests,
timeouts, incorrect wheel provenance and silent objective fallback fail the
command. Results are written before validation so failed jobs retain evidence.
An image-build failure before the local entrypoint starts is reported by Modal
itself and must be recorded separately; it is not a completed GPU test.

The smoke uses one T4, two CPU cores, 8 GiB requested memory, one container,
no application retries, a 300-second remote function timeout and a 240-second
pytest subprocess limit. Platform startup/build/restarts are outside this
execution timing. No cost or scaling claim follows from the smoke.

The two mandatory cases check installed wheel contents, CuPy/Numba pointer
sharing and owner lifetime with a real kernel, and a two-round Normal fit
whose gradients and native tree calls actually run on device. Private imports
and spies here are test instrumentation, not the public extension examples
planned for P6. The small dataset is training-only; NLL is a finite-value
sanity check, not evidence of held-out quality or complete CPU/CUDA parity.

Offline revalidation:

```bash
uv run python -m benchmarks.foundation.runner benchmarks/results/foundation/RUN_ID
```

Refresh the hash-locked requirements only when changing the environment:

```bash
uv export --locked --extra cuda --extra test --no-dev --no-emit-project --prune jax --prune jaxlib --no-annotate --no-header -o benchmarks/foundation/requirements.txt
```

Markers remain in the export and are evaluated on Linux. JAX is not required
for these NumPy/CuPy/Numba smoke cases. Record the exact installed versions in
each run. The CUDA base digest is Linux amd64 CUDA 12.4.0 devel Ubuntu 22.04,
resolved from NVIDIA's registry; the image's Python patch version is recorded
at runtime. The uv image installer is pinned to 0.12.1.

P2 weighted correctness regression (includes the smoke cases):

```bash
uv run python -m benchmarks.foundation.prepare --suite correctness
uv run modal run benchmarks/foundation/modal_app.py::foundation_correctness
```

This currently covers fixed-bin weighted histograms/Newton predictions and
three-round weighted Normal/Poisson CPU/CUDA comparisons. It does not yet
constitute the complete P2 baseline gate.

## Remaining P2 execution boundaries and baseline

The `boundaries` suite adds eight real-device tests to the five correctness
cases: custom/exposure/generic fallback, device-error rollback, row/column
sampling preflight and Normal/Poisson callback/eval/cross-device persistence.

```bash
uv run python -m benchmarks.foundation.prepare --suite boundaries
uv run modal run benchmarks/foundation/modal_app.py::foundation_boundaries
```

The `baseline` suite includes all 13 boundary/correctness cases and stops on
any failure before entering its real-data matrix. Download the public
California Housing archive once into the ignored data directory:

```bash
uv run python -c 'from benchmarks.foundation.dataset import fetch; fetch("build/foundation_data/cal_housing.tgz")'
uv run python -m benchmarks.foundation.prepare --suite baseline
uv run modal run benchmarks/foundation/modal_app.py::foundation_baseline
```

The archive hash is sklearn 1.8.0's published hash, and the transformation was
checked against that installed sklearn loader. `housing.json` freezes the
archive/array/split hashes. Float32 conversion follows the original per-row
ratio transformations; no learned scaling is applied. Each seed 0/1/2 uses a
60/20/20 train/validation/test permutation. Only training data fits bin edges.

The predefined Normal model uses 30 rounds, depth 3, learning rate .05 and 64
bins. For every CPU/CUDA × seed × no-eval/eval cell, a fresh Python subprocess
and empty NUMBA_CACHE_DIR measure first and repeated fit. Fits include binning,
objective math, copies and compilation; imports, dataset loading and container
startup are excluded. Prediction includes test binning. Small path-counting
wrappers are included in timings. Repeated CUDA predictions use rtol=2e-5 /
atol=2e-6 because floating-point atomic reductions need not be bit-identical.

CPU/CUDA quality gates are frozen before collection: per seed/mode NLL absolute
difference <= .01 * max(1, abs(CPU NLL)), CRPS regression <= 1%, and coverage90
absolute difference <= .01. These real-data gates cannot override the strict
micro-oracle tests. Three seeds do not establish statistical significance.

The baseline function is limited to one T4, two CPU cores, 8 GiB requested
memory, no retries and 1800 seconds. Its pytest subprocess is capped at 1740
seconds and each matrix worker at 150 seconds. Failure output and partial
completed cells are retained. Exact CPU model is recorded if /proc exposes it.
The trainer transfer counter is partial, and is not total PCIe traffic or a
zero-transfer assertion. GPU memory peaks and scaling remain later gates.

Status: P2.2/P2.3 completed on real T4: 14 passed, no skips, all 12 baseline
cells completed. See [raw evidence and scoped timing/quality results](../results/foundation/20260905T084129Z-2574e387/README.md).
First-fit timings do not clear CUDA driver caches; the container does not
expose its physical CPU model. These limits are recorded in the artifact.

P4.1 histogram validation uses `prepare --suite histograms` and Modal entrypoint
`foundation_histograms`. It uploads the independent CPU test module as
`histogram_oracle.py`, runs the existing two smoke cases plus one comprehensive
batch histogram device case, and requires all three without skips. The device
case covers small exact, random and empty aggregates; this is not a GPU extension
trainer or performance benchmark.

P4.2 uses `prepare --suite splits` and entrypoint `foundation_splits`, adding
`test_splits.py` and the CPU row-mask oracle (`split_oracle.py`). It reruns the
two smoke cases and histogram case, then verifies split/gain/routing, exact ties,
invalid routes, and rebuilding child histograms from real routed rows. All four
GPU cases and their evidence fields are required; no skips satisfy the gate.

P4.3 uses `prepare --suite leaves` / `foundation_leaves`. The allowlist adds
`test_leaves.py` and `leaf_oracle.py`; all five smoke/histogram/split/leaf cases
must pass. Leaf evidence checks direct row sums, empty/zero-curvature behavior,
plugin output/ownership errors and two rounds where clipping changes the next
weighted gradient. The GPU two-round check composes primitives; it is not an
assembled experimental GPU Booster validation.

P4.4 uses `prepare --suite builder` / `foundation_builder`, adding `test_builder.py`
and the whole-tree row oracle `builder_oracle.py`. Six required cases include
prior primitives, whole-tree topology/leaf checks, input-view lifetime,
compact-only finalization, two-round/two-channel Normal composition at 16 and
4,097 rows, bounded rules, and CPU inference after persistence. Synthetic
NLL/CRPS check numerical agreement; this is not a product-quality benchmark.


`--suite trainer` / `foundation_trainer` extends the seven-case suite to actual
strict experimental Booster.fit. It checks default/external dispatch, weighted
scheduled Normal parity and CPU persistence/rollback, all Normal/Poisson ×
ordinary/natural adapter modes and invalid-statistic rejection. Counts and
reports distinguish compact downloads, CPU initialization/binning and device
input copies. The frozen run has no nsys executable: profiler evidence is open.
See `results/foundation/20260905T180000Z-7d73ba83` under benchmarks for raw evidence.


`--suite extensions` / `foundation_extensions` installs the two independent
example wheels alongside OpenBoost. The three-case suite checks installed module
contents, independent GPU mathematics, eight CPU/CUDA composition cells and a
standalone GPU demo, then removes the plugins and checks nine exact CPU model
roundtrips in a new interpreter. `20260905T181351Z-1aee9568` is the passing raw
artifact; the initial standalone-script failure is retained separately.

## P7 resident value protocol

`prepare --suite value` freezes the P2 Housing archive, splits, configuration and
raw baseline plus both independent 0.2.0 wheels. Run
`uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_value`.
One T4 (2 CPU, 8 GiB, 1800-second function limit, no retry) executes seeds 0/1/2
and legacy CPU, legacy CUDA, strict experimental CUDA, independent A+B+C CUDA.
Each cell uses a fresh subprocess and fresh Numba/CuPy cache directories, one
process-first fit and three warm fits. Imports, data loading and context startup
are excluded; binning, gradients, transfers and fit compilation are included.
The driver cache is not cleared: this is not a machine-cold benchmark.

Compare the default candidate with the same-run legacy CUDA reference, and check
that reference against frozen P2 held-out quality. Historical first-fit timings
have a different cache policy and are not a direct latency comparator. Require
NLL difference <= 0.01 * max(1, abs(reference)), CRPS <= 1.01 * reference and
coverage90 difference <= 0.01 on every repeat/seed. Report a regression instead
of rejecting its artifact. A ratio of cross-seed median warm fit times above
1.2 triggers design review. The independent Fisher/bounded/scheduled algorithm
uses bound=0.5 and tau=1 without tuning; its math differs and its quality/cost
is reported separately. Strict CUDA eval/callbacks remain unsupported; the P2
eval cells have no strict-GPU performance counterpart.

A separate fifth fit records cProfile host attribution and named transfer wrappers;
a sixth memory-only fit samples device-wide used memory every 5 ms. Inclusive times and nested
wrapper counts overlap; they are not kernel times or a complete transfer audit.
Sampled memory is a lower bound including contexts and allocator caches, not an
exact per-fit peak. CuPy pool figures exclude Numba allocations. A CUDA trace is
still a separate evidence gap. Report T4 seconds, not inferred billing dollars.


The original P7 value run used concurrent profiling/sampling; its cProfile times
were contaminated by the sampler and remain retained with that limitation.
`prepare --suite value_profile` / `foundation_value_profile` collect only seed 0
for the four strategies, one compile warmup then isolated host and memory fits.
The 600-second diagnostic job does not replace the original timing matrix.
It checks quality against that committed parent result, carries its hash, and
adds synchronized inclusive boundary timers. Nested timers overlap and include
synchronization overhead; they are diagnostic, not production latency claims.


P7 outcome: [original resident matrix](../results/foundation/20260905T183820Z-3c245f2d/README.md)
passes default quality but fails the performance budget (13.899x legacy CUDA fit
median). The [isolated diagnostic](../results/foundation/20260905T184856Z-dcd49569/README.md)
points to tree construction and its boundary as the dominant cost. Neither run
establishes a general GPU speed/cost advantage or external adoption.


[Fixed-slot growth follow-up](../results/foundation/20260905T193308Z-5ebd75ab/README.md)
retains all checks and records default warm fit 1.868 s versus the original
2.079 s. The paired legacy ratio is still 12.888x: quality passes, performance
budget fails. CUDA correctness and independent CPU wheel evidence accompany it.
