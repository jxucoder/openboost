# Run 11: Parallel field validation passes correctness and cost gates

All **474 cases pass** on real T4 at clean revision
`dd842478a1af55a88cee580682f9471342f1e404`. All three frozen cost gates pass, with
byte-identical original/candidate models and exact predictions across every GPU
repetition. All 28 completed CPU/GPU fits replay on the offline macOS host from
the retained lossless input bytes; independent task scores also verify. The one
approved invocation is consumed, with no retry.

The [raw verdict](verdict.json), [manifest](manifest.json) and
[derived audit](analysis.json) retain exact cases, sources, settings and failures.
The preceding run-10 failure and partial timings remain unchanged in their archive.

## Complete timing results

Times below are unprofiled fit seconds. Each GPU arm has one first and three warm
fits; each CPU control has one first and one warm. The reported GPU warm value
is the median of all three repetitions.

| Workload | Original GPU first / warm | Candidate GPU first / warm | Warm time reduction | Frozen gate |
| --- | ---: | ---: | ---: | --- |
| Squared, 10,000 rows | 8.580 / 2.838 | 8.058 / 2.749 | 3.15% | Pass: at most 10% regression |
| Squared, 100,000 rows | 18.465 / 13.513 | 14.230 / 8.947 | 33.79% | Pass: at least 20% reduction |
| Normal, 10,000 rows | 18.341 / 6.538 | 17.444 / 5.665 | 13.35% | Pass: at most 10% regression |

These compare the original and candidate OpenBoost implementations on identical
generated inputs and the same T4 allocation. All settings stay at sixteen features,
32 bins, depth three, twenty fixed rounds, learning rate 0.1, regularization 1
and seed 7. Normal uses joint Fisher updates and forty scalar terms. Data includes
weights, zero-weight rows, offsets and missing features; holdout size is one fifth
of training size. This is one synthetic seed, with fixed CPU/original/candidate
order and no confidence interval or external-library comparison.

| Squared workload | CPU first / warm (s) | CPU / candidate GPU warm ratio | Relative half-MSE difference | Normalized prediction RMSE |
| --- | ---: | ---: | ---: | ---: |
| 10,000 rows | 6.613 / 7.826 | 2.847 | 1.090e-7 | 1.947e-7 |
| 100,000 rows | 14.996 / 14.962 | 1.672 | 3.537e-8 | 3.455e-7 |

Both CPU comparisons complete and remain well below the frozen 1% quality and
prediction thresholds. The fresh-process candidate GPU first fit at 10,000 rows
is slower than the CPU first fit; the 2.847 ratio applies to warm execution.
No Normal CPU fit is part of this packet. No XGBoost, LightGBM or CatBoost speed
parity, general GPU advantage or formal E4 pass follows from these measurements.

## What changed and what remains costly

The candidate changes only the field-validation kernel and its launch size in
`device.py` and `_device_kernels.py`: one cooperating 128-thread block per column
replaces one serial thread per column. It preserves all row checks, integer flags,
allocation ownership, launch counts and numerical algorithm decisions. All 39
new validation cases pass, including empty/tail shapes, float32/float64, nonfinite
and negative values, zero-weight invalid rows and injected failure recovery.

The separate 100,000-by-two-field profile runs one first and ten warm calls per
arm. Median warm CUDA-event interval falls from **15.189 ms to 0.507 ms**, about
30x. Median wall time falls from 15.197 ms to 0.514 ms. These intervals include
host enqueue gaps and blocking flag checks; they are operation measurements,
not exclusive kernel time and not eligible for fit ratios.

Large squared training still takes 8.947 warm seconds. Each squared fit retains
4,531 launches and 6,272 synchronizations; Normal retains 9,417 and 12,982. Last-fit
host dispatch time is about 1.30 seconds for squared and 2.69 seconds for Normal.
Those observations show remaining overhead but do not identify the next dominant
kernel. Row-index validation, histogram and numerical-loss reductions are unchanged.
All 24 GPU fits and both profiles end with zero owned live bytes. The largest
recorded candidate squared private pool peak is 14,123,008 bytes, below 512 MiB;
this is not total driver/device memory usage.

## Evidence, environment and replay

All 88 uploaded source hashes match the clean execution revision and worker
snapshot. All 31 installed candidate files match. A separately built original
wheel matches all 31 run-10 core hashes, with only two original-file overlays.
Both installations verify the same eighteen pinned packages. Installed D2 source
and version checks and the separate NumPy/core-only CPU environment also pass.

The worker reports Tesla T4, 15,360 MiB, driver 580.95.05, runtime/driver API
12090/13000, Python 3.12.1, NumPy 2.3.5, Linux 4.19.0/gVisor/x86_64/glibc 2.35.
Two CPU cores and 8,192 MiB are requested; eighteen guest-visible CPUs are not
the CPU entitlement. Each timing child sets BLAS/OpenMP thread limits to one
and uses private CuPy/driver caches. Fit time includes context, host binning,
training, model export and cleanup, including triggered JIT. Input loading,
post-fit replay/scoring/prediction and serialization consume child wall time
outside the fit timer. First/warm differences are not pure compilation cost.

All seventeen declared JSON artifacts are retained, totaling **33,985,980 bytes**,
within the 64 MiB cap. Twenty raw artifact hashes plus the manifest binding are
verified. No repetition times out. Worker time is 380.143 seconds and pytest time
377.754 seconds, below the 900/600-second caps. Total dispatch time, including
image construction/setup, is 765.509 seconds. No retries or additional GPU
invocations occur.

The local audit uses macOS 26.3 x86_64, Python 3.12.12 and NumPy 2.3.5. It loads
saved array bytes and verifies Problem identities, every model/prediction, task
metrics, paired decisions, deadlines, source provenance and the original verdict.
No input regeneration is needed. This resolves run 10's cross-platform input
reconstruction gap for these new artifacts without changing the old evidence.

With the executed core and helpers checked out, run:

```bash
OPENBOOST_BACKEND=cpu uv run --no-sync python benchmarks/v1/evidence/parallel-validation-105/analyze.py --check
```

The analyzer refuses later core/helper changes; use the recorded execution revision
in an isolated checkout for historical replay. The archive index binds every
retained file except the index itself. Existing regression tests retain JUnit
outcomes, not a new complete trajectory archive. The omitted 96-case Normal runtime
matrix and two old lowering/cost cases are not relabeled as run-11 passes.

## Decision

Keep the optimization: its exact-model checks and the predeclared large-workload
cost gate pass. Stop at the planned retrospective. Next construction should return
to Sprint 080's required CUDA recipe cells, then compatible train-many and formal
real-data cost evaluation. Further kernel optimization needs evidence identifying
a remaining cost or a concrete consumer blocker. All required R/C/A scope remains;
full Normal conformance, author/adoption evidence and formal E4 are not established.
