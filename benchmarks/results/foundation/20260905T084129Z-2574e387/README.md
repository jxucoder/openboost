# P2 execution boundaries and real-data baseline

Source `99621ae9e640be9c1144a015a39de43d3bf50ae9`, clean; wheel SHA-256
`40a5ec42661eba6f72bf340147f29d0fd2ffed37877e77b6981d78cd1d0f888b`.

**14 passed, 0 skipped**, pytest 104.58 s; remote function wall time 108.88 s.
The CLI and offline evidence validator both returned success. Raw records:
[manifest](manifest.json), [results](results.json), [JUnit](junit.xml).

## Correctness and execution boundaries

All five earlier smoke/weighted cases and eight new boundary cases passed:

- Same-name custom distribution, exposure and generic-tree fallback are visible
  and match CPU predictions on their controlled fixtures.
- A deliberate device-kernel error propagates and restores unfitted trainer
  state; unsupported GPU row/column sampling fails before binning/updates.
- Normal and Poisson callback/eval values and unweighted CPU/CUDA predictions
  match; GPU-save/CPU-load and CPU-save/GPU-load preserve predictions.
- The callback fixture downloads raw scores at the observed trainer boundary
  once per round (3 times); the no-callback fixture has no such downloads.

## Frozen dataset and quality

California Housing, 20,640 rows, 8 features, target in units of 100,000 USD.
Original archive SHA-256:
`aaa5c9a6afe2225cc2aed2723682ae403280c4a3695a2ddda4ffb5d8215ea681`.
Float32 feature/target arrays SHA-256:
`34fc72e53a6329a89a9ae792e92f37853cea98adfbc6feea28cc08dc06f68ca1`.
All source, copied input, dataset and split hashes were independently checked.

Seeds 0, 1, 2 use fixed 60/20/20 splits: 12,384 training, 4,128 validation,
4,128 test rows. Only training data fits the 64-bin feature boundaries.
Normal model: 30 rounds, depth 3, learning rate .05, min_child_weight=1,
reg_lambda=1 and no sampling. No target tuning or learned scaling.

The following are held-out repeated-fit results without eval. Eval-mode CPU
metrics are identical; GPU eval-mode absolute differences from CPU are at most
1.34e-8 NLL, 1.12e-8 CRPS and zero coverage difference. Every seed/mode passes
the predeclared design gates; no threshold was changed after collection.

| Seed | CPU NLL | CUDA NLL | CPU CRPS | CUDA CRPS | CPU/CUDA coverage90 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.094473764 | 1.094473769 | 0.398590950 | 0.398590956 | 0.967781008 |
| 1 | 1.113434585 | 1.113434586 | 0.410066944 | 0.410066944 | 0.963905039 |
| 2 | 1.096392043 | 1.096392045 | 0.403819868 | 0.403819868 | 0.964389535 |

CPU repeated predictions are bit-identical; maximum CUDA repeated-parameter
absolute difference is 4.768e-7, within the declared tolerance. These are
parity results for a fixed, untuned baseline. Nominal 90% intervals cover
96.39–96.78%, so this run does not establish good calibration or tuned quality.
Three seeds are not evidence of statistical significance.

## End-to-end timing baseline

Medians across the three seeds, seconds. Each backend/seed/mode uses a fresh
Python process and NUMBA_CACHE_DIR, then a first and repeated fit (12 processes,
24 fits total). Fits include binning, gradient math, copies, JIT compilation
and path-counting instrumentation. Test prediction includes binning and all
parameter channels. Imports, data loading, image build and container startup
are excluded. CUDA driver caches were not explicitly cleared: “first fit”
means process-first under this protocol, not a machine-cold timing.

| Backend | Eval mode | First fit | Repeated fit | Prediction after repeated fit |
| --- | --- | ---: | ---: | ---: |
| CPU | none | 4.413173 | 2.401469 | 0.684316 |
| CUDA | none | 2.412475 | 0.145193 | 0.067331 |
| CPU | validation | 5.106433 | 3.026923 | 0.676538 |
| CUDA | validation | 2.450300 | 0.230342 | 0.016565 |

Evaluation invokes tree prediction during fitting and populates per-tree GPU
array caches; no-eval native training updates raw scores directly. These modes
therefore reach test prediction with different cache states. The table retains
mode-specific timings. This is one bounded baseline run with small repeated
samples, not a general performance claim against another boosting library.
Future comparisons must use the same protocol and matched quality.

All CUDA fits execute 30 device objective steps and 60 native trees without
fallback warnings. The partial trainer device-to-host counter is 0 without
eval and 60 with eval. It excludes compact tree conversion and backend-internal
copies and must not be interpreted as total PCIe traffic or zero transfers.

## Environment and reproduction

Tesla T4 15,360 MiB, driver 580.95.05; CuPy runtime 12.9, driver API 13.0,
pinned CUDA 12.4 toolkit base. Two requested CPU cores, 8 GiB requested memory,
thread counts fixed at 2. `/proc/cpuinfo` reports CPU model `unknown`; the
physical CPU model is therefore unavailable. Full OS, package and resource
metadata are in `results.json`. No peak-memory/scaling claim is made.

From the tested source in a clean checkout:

```bash
uv run --no-sync python -c 'from benchmarks.foundation.dataset import fetch; fetch("build/foundation_data/cal_housing.tgz")'
uv run --no-sync python -m benchmarks.foundation.prepare --suite baseline
uv run --no-sync modal run benchmarks/foundation/modal_app.py::foundation_baseline
```

Offline validation:

```bash
uv run --no-sync python -m benchmarks.foundation.runner benchmarks/results/foundation/20260905T084129Z-2574e387
```

The single-T4 function has timeout 1800s, retry=0 and max_containers=1; pytest
is limited to 1740s and each matrix process to 150s. All 13 prerequisite cases
execute before the baseline matrix, with maxfail=1.
