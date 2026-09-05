# Fixed-slot growth: modest improvement, performance budget still fails

Clean measurement source `a5bf26d3d54374e4c524375a02294f0c64abbc1c`, implementation
`81acf0b`. Core wheel `8c94b62773f6541a983280b5970642de46c60a35e2bed0fcdf62e9d4a9cb3077`.
T4: **3 passed / 0 skipped**, all 12 cells completed without fallback. This is
complete evidence, not a passing performance gate. Same P2 Housing hash/splits,
seeds 0/1/2, 30 rounds, depth 3, learning rate .05 and 64 bins as original P7.

| Strategy | Seed 0 warm fit s | Seed 1 | Seed 2 | Cross-seed median s |
|---|---:|---:|---:|---:|
| Legacy CPU | 2.3427 | 2.3476 | 2.3655 | 2.3476 |
| Legacy CUDA | .1444 | .1476 | .1449 | .1449 |
| Default experimental CUDA | 1.8680 | 1.8537 | 1.8722 | 1.8680 |
| Independent A+B+C CUDA | 1.8870 | 1.9187 | 1.8895 | 1.8895 |

Against the [original P7 matrix](../20260905T183820Z-3c245f2d/README.md), recorded
default warm fit median decreased from 2.078694 to 1.868019 s (10.13%). Legacy
CUDA also decreased from .149556 to .144938 s. The ratio against each run's
paired legacy reference changes from 13.899x to **12.888x** (7.27% lower).
Per-seed current ratios are 12.935, 12.555 and 12.917. These are three-split
observations across separate T4 runs, not a significance claim or attribution
of every percentage point to code. Keep the small fixed-shape implementation;
it preserves checks and shows consistent raw reductions, but does not solve G4.

Process-first default fits: 19.488, 19.419, 19.369 s. Default prediction medians:
.7411, .7536, .7456 s versus legacy CUDA .0699, .0701, .0720 s. The experimental
API still predicts on CPU. No prediction speed improvement is claimed.

| Seed | Default NLL | CRPS | Coverage90 | Independent A+B+C NLL | CRPS | Coverage90 |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.094474 | .398591 | .967781 | 1.483626 | .599585 | .923934 |
| 1 | 1.113435 | .410067 | .963905 | 1.511263 | .618114 | .915698 |
| 2 | 1.096392 | .403820 | .964390 | 1.494126 | .606814 | .922481 |

Every default fit/seed passes the original quality thresholds, and same-run
legacy matches frozen P2. Nominal 90% coverage remains overconservative. The
independent Fisher/bounded/scheduled algorithm still has worse proper scores;
closer nominal coverage alone is not a win. No seed, configuration or tolerance
was changed. Strict CUDA eval/callbacks remain outside this comparison.

## Remaining overhead

Isolated seed-0 diagnostic wall time is 2.0466 s: tree/session boundary 1.9099 s,
nested LevelWiseBuilder 1.3641 s, objective boundary .0866 s. The preceding
[isolated profile](../20260905T184856Z-dcd49569/README.md) recorded builder 1.5880 s
and tree/session 2.1226 s. Boundary timers are synchronized, nested and include
instrumentation overhead; do not use them as uninstrumented production timings.
Removing split-array compaction addresses only part of the overhead. Remaining
candidates include repeated validation/synchronization and the separate cached
prediction traversal; none has been removed or shown to deliver a speedup here.

Four timed fits per cell precede an isolated fifth host-profile fit and sixth
memory-only fit for CUDA (CPU memory phase is inapplicable). The original manifest's
profile prose predates this separation; the hashed worker and each memory scope
record actual execution. `d816954` clarifies future manifests without rewriting
this artifact. Timed fits use fresh process/Numba/CuPy caches, exclude imports,
data loading and context startup, include binning/gradients/transfers/JIT, and
leave the driver cache intact. They are not machine-cold timings.

Sampled device-wide memory is a lower bound including contexts/caches, not an
exact per-fit peak; named transfer wrappers are not a full CUDA trace. nsys is
unavailable. Function 294.95 s / pytest 291.04 s are not billing records. One
T4, requested 2 CPU / 8192 MiB, threads 2, 1800-second limit, no retries. Full
OS/Python/package/GPU/driver/data/hash/command provenance is retained in JSON.

[Real CUDA correctness](../20260905T193001Z-cece3e8f/README.md): 7 passed / 0 skipped,
including Normal/Poisson and CPU/CUDA task metrics. [Independent CPU wheels](../p7-fixed-slot-cpu-a5bf26d/README.md):
7 passed plus six exact predictions after plugin removal. Both verify this same
core wheel. No validation, persistence, sampling or feature capability weakened.

Reproduce from the measurement source revision:

```sh
uv run --no-sync python -m benchmarks.foundation.prepare --suite value
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_value
```

Validate offline:

```sh
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T193308Z-5ebd75ab
```

G4's 1.2 fit-ratio budget still fails. Keep the bounded experimental research API;
do not replace the legacy CUDA path or claim general speed/cost superiority.
External adoption, exact per-fit GPU peak and a complete CUDA trace remain open.
