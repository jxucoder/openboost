# P7 isolated host profiling follow-up

Clean source `9bb1ff38cce08df5631cd0aa7fcd3a84b20cf8cd`. T4: **3 passed / 0 skipped**.
Four seed-0 strategies, one compilation warmup, then a host-profile fit and a
separate memory-only fit. Quality matches the corresponding original cells.
The [original 12-cell timing matrix](../20260905T183820Z-3c245f2d/README.md) remains
unchanged; this diagnostic run neither replaces its times nor changes its failed
1.2 performance budget. Its results hash is recorded in profile_parent.

## Synchronized inclusive diagnostics

| Strategy | Profile fit wall s | Objective boundary s | Tree/session boundary s | LevelWiseBuilder inside session s |
|---|---:|---:|---:|---:|
| Legacy CPU | 2.3961 | .0146 | — | — |
| Legacy CUDA | .2448 | .0200 | .1751 (native trees) | — |
| Default experimental CUDA | 2.2601 | .0885 | 2.1226 | 1.5880 |
| Independent A+B+C CUDA | 2.2135 | .1045 | 2.0674 | 1.5527 |

Each objective runs 30 times and each tree path 60 times. Nested timers overlap:
the .0348 s underlying built-in objective is already inside the default's .0885 s
objective boundary, and LevelWiseBuilder is inside the tree/session boundary.
Synchronization and cProfile add overhead; these are not production fit times.

The default spends about 94% of this diagnostic wall time in its tree/session
boundary, with about 1.588 s inside the builder and .535 s elsewhere in that
boundary. Objective arithmetic is not the dominant cost. The no-sampler profile
shows 2,400 CuPy any calls and 2,190 ndarray all calls on this path, along with
180 histogram/split/partition calls and compact-tree validation. These counts
support investigating repeated validation and synchronization, but do not prove
which CUDA kernel or check would deliver a particular speedup.

Source inspection shows device copies of borrowed inputs, scalar validation,
compact tree snapshots, and independent cache-validation traversal. Keep those
correctness contracts. First investigate batching/reusing validation or reducing
redundant traversal/synchronization with independent parity tests; do not simply
remove checks or route an explicit external builder through the legacy builder.
Only rerun affected comparisons after an actual change; no such core optimization
is included in this evidence. This is a design review, not a claimed fix.

## Transfer and memory boundary

Each CUDA profile records 300 named compact D2H wrappers: Numba copy_to_host for
legacy, CuPy asnumpy for the experimental paths. Default asnumpy cumulative host
time is .0067 s. Named wrappers exclude internal/scalar traffic and are not a
complete transfer audit. No CUDA trace was captured; nsys is unavailable.

Memory now runs without any host profiler in a separate warm fit. Default
experimental device-wide sampled usage starts at 332,333,056 bytes and reaches
334,430,208 bytes (2 MiB delta). Legacy and A+B+C show zero sampled delta. Those
figures include contexts, allocator caches and other device allocations; a 5 ms
sample can miss peaks. Zero delta does not mean no training allocation, and this
is not an exact per-fit peak. No GPU billing/cost dollars are inferred.

The original combined sampler/profile had inconsistent inclusive attribution.
This follow-up removes memGetInfo/sampler-lock contamination and adds explicit
synchronized boundary timers. Retain the original profiles with their limitation;
do not use their time percentages for causal claims.

Core and both plugin wheel hashes exactly match P6 and original P7. Python
3.12.1, Tesla T4, driver 580.95.05, CUDA runtime 12090; full package/environment
and dataset provenance in manifest/results. Requested 2 CPU / 8192 MiB, threads 2,
CPU model unknown. Pytest 79.98 s; remote function 84.03 s. One T4, 600-second
limit, no retries. These function times include warmup/profiling, not billing.

Reproduce from the source revision:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite value_profile
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_value_profile
```

Validate offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T184856Z-dcd49569
```

Verdict unchanged: default quality passes, GPU performance budget fails, G5
external adoption unverified. Retain the experimental research API; do not replace
the legacy CUDA path or claim a generally superior GPU foundation.
