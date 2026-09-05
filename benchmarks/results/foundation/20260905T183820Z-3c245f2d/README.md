# P7 resident value: quality passes, performance budget fails

Source `eb6121868a0ddcc25d4aae7c108e331e0ff52f4b`, clean wheel-only T4 run.
3 tests passed / 0 skipped; all 12 cells completed without fallback. This means
complete evidence, **not** a passing performance gate. Housing seeds 0/1/2,
12,384 train / 4,128 validation / 4,128 test rows, eight features, 30 rounds,
depth 3, learning rate .05, 64 bins; exact data/config and environment in manifest.

| Strategy | Seed 0 warm fit s | Seed 1 | Seed 2 | Cross-seed median s |
|---|---:|---:|---:|---:|
| Legacy CPU | 2.4404 | 2.4481 | 2.5764 | 2.4481 |
| Legacy CUDA | .1605 | .1478 | .1496 | .1496 |
| Experimental default CUDA | 2.0855 | 2.0787 | 2.0733 | 2.0787 |
| Independent Fisher + bound + schedule CUDA | 2.1148 | 2.0735 | 2.0679 | 2.0735 |

Each warm cell is the median of three fits after a process-first fit, all in an
independent process with fresh Numba/CuPy cache directories. The default candidate
is **13.899x** legacy CUDA by the preregistered cross-seed median ratio, well above
the 1.2 budget. Per-seed ratios are 12.995, 14.063 and 13.863. Experimental CPU
prediction also costs about .73–.74 s versus .068–.070 s for legacy GPU prediction.
This is an end-to-end API comparison with different prediction execution devices.

Process-first fit ranges: CPU 4.440–4.589 s, legacy CUDA 2.304–3.283 s,
experimental default 19.655–19.915 s, independent extension 20.649–21.012 s.
Imports/data loading/device context startup are excluded, driver cache was not
cleared. These are not machine-cold times. Same-run legacy timings are the
comparator; older P2 timings have a different cache policy.

| Seed | Default candidate NLL | CRPS | Coverage90 | Independent A+B+C NLL | CRPS | Coverage90 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.094474 | .398591 | .967781 | 1.483626 | .599585 | .923934 |
| 1 | 1.113435 | .410067 | .963905 | 1.511263 | .618114 | .915698 |
| 2 | 1.096392 | .403820 | .964390 | 1.494126 | .606814 | .922481 |

Default quality passes on every fit/seed, and same-run legacy agrees with frozen
P2. This is numerical agreement, not good calibration: nominal 90% coverage is
96.4–96.8%. A+B+C changes curvature, clips leaves at .5 and uses tau=1 decay;
it has substantially worse proper scores here. Its closer coverage alone is not
a quality win. No tuning, seed deletion or threshold changes occurred. Strict
CUDA eval/callbacks remain unsupported and have no matching performance row.

## Profiling limitations and follow-up

Separate fifth fits collected host cProfile, path calls, named transfer wrappers
and 5 ms sampled device-wide memory. Path counts verify 30 objective / 60 trees;
legacy records 300 copy_to_host calls, experimental 300 asnumpy calls per fit.
Nested/internal/scalar copies are not a complete transfer audit. GPU memory
peaks sampled after warmup can equal the initial context/allocator footprint:
zero delta **does not mean zero training memory**.

The sampling thread appears in the returned cProfile attribution (memGetInfo and
thread lock time); candidate parent/child cumulative times are also inconsistent.
Do not infer a causal time breakdown or an optimization target from those time
percentages. Original profiles remain verbatim. Follow up with a separate host
profile without a concurrent memory sampler; keep these timed fits unchanged.
No CUDA kernel trace was captured, and sampled memory is not an exact peak.

T4, driver 580.95.05, runtime 12090, Python 3.12.1, NumPy 2.3.5, CuPy 13.6.0,
Numba 0.63.1 / numba-cuda 0.27.0. CPU requested 2, RAM 8192 MiB, threads 2;
CPU model not exposed. Pytest 293.30 s, remote function 297.30 s. Function time
includes initialization, processes, JIT and profiling; it is not a billing record.
Main and both plugin wheel hashes match P6 exactly (full hashes in manifest).

Reproduce from the source revision:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite value
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_value
```

Validate the retained evidence:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T183820Z-3c245f2d
```

Decision: retain the experimental API for research; do not replace the legacy GPU
path or claim speed/cost superiority. G3 technical extension capability holds;
G4 has a negative performance result plus profiling/peak-memory gaps; G5 is open.


[Isolated follow-up](../20260905T184856Z-dcd49569/README.md) subsequently passed
with unchanged quality and separate host/memory fits. It localizes most diagnostic
time to the tree/session boundary. The original timings and raw profiles above
remain unchanged.
