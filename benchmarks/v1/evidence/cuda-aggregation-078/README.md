# 078-A: Real T4 named-field and routed-histogram result

Clean source `ad2f4e6`. All 33 installed-wheel tests pass: 21 aggregation/identity/
failure checks and twelve storage regressions. The exact JUnit case set, all 35
snapshot source hashes, 23 installed production modules, 17 pinned package versions
and three result-artifact hashes verify. There are no missing or skipped cases.
See manifest.json, verdict.json, junit.xml and pytest.log in this directory.

## Verified behavior

- Once-only objective weighting and independent named information, including zero
  objective weights and appended/renamed nonnegative D2 information columns.
- Original-row histogram sums and integer counts, unsorted/empty/zero-mass/all-
  missing subsets, variable feature bin counts and zero padding.
- Resident int32 selections, duplicate/out-of-range rejection, owned versus borrowed
  lifetime behavior, foreign/forged/stale records and shape/role/nonfinite rejection.
- Allocation failure after a partial histogram allocation, recovery with valid
  inputs, weighting overflow rejection and owned-stream restoration.
- The frozen 8192x32 float32 fixture, 32 regular bins and four named fields, for
  all rows and every third row. Float64 original-row loops are the independent
  oracle; float32 uses rtol=1e-4/atol=1e-5 and counts/positions compare exactly.

Each four-field histogram call has no host upload or bulk result download. It
exports 32 bytes of compact finite-validation flags, checked separately from
diagnostic row/sum/count/total exports. The tested kernels run through numba-cuda
on the owned CuPy stream; outputs and validation scratch use the private pool.
The device supports additive information here, not D2 candidate selection yet.

## Resource observations and limits

| Largest-shape context | All rows | Every third row |
| --- | ---: | ---: |
| Uploaded bytes, including explicit preparation/field/row inputs | 1507328 | 1485484 |
| Exported bytes, including reference diagnostics | 58192 | 36348 |
| Validation-export bytes within that total | 64 | 64 |
| Peak live logical bytes | 1663776 | 1641932 |
| Sampled private-pool peak bytes | 1665024 | 1643520 |
| Explicit synchronization count at logged checkpoint | 21 | 21 |

The private pool cap is 16 MiB. Its sampled peak excludes driver/context/module/JIT
and outside allocations; it is not whole-device peak memory. The log retains
31 low-occupancy warnings from small kernel grids. Host dispatch timing includes
lazy JIT and does not measure GPU elapsed time. Pytest reports 4.37 seconds and
the worker reports 5.38 seconds; neither is an end-to-end boosting benchmark.
No cost or speed gate is evaluated.

## Environment and consumed budget

Tesla T4, 15360 MiB, driver 580.95.05; Python 3.12.1; Linux 4.19/gVisor x86_64,
glibc 2.35. The function requests two CPUs and 8192 MiB; os.cpu_count reports 18
visible logical CPUs, which is not allocated CPU capacity. Package versions are
fully listed in the manifest, including CuPy 13.6.0, numba-cuda 0.27.0,
Numba 0.63.1, llvmlite 0.46.0 and NumPy 2.3.5. The imported CUDA module comes from
the installed numba_cuda package.

The image tag is CUDA 12.6.3-devel, while the loaded runtime reports 12090 and
driver API reports 13000. Preserve those distinct observations. The exact image,
CLI, test command, source/configuration hashes and timestamps are in manifest.json.

This is run 2 of the two-run 085 allowance: one T4 invocation capped at 900 seconds,
600-second tests, zero retries. Both approved runs are now consumed. Do not rerun
the harness without a new declared allowance. The reproducing command was:

```bash
uv run --no-sync python -m benchmarks.v1.cuda_aggregation_preflight /tmp/openboost-cuda-aggregation-078-run2
```

## Reflection and next boundary

078-A passes its declared real-device aggregation acceptance. Public named fields
retain weight/identity semantics, and row reductions stay on CUDA with explicit
validation costs. This supplies reusable primitives for the next split operations.

078 overall remains open: candidate enumeration/selection, routing by a selected
condition, leaf solves, device gradients, accepted/proposal integration, two-round
training and trained CPU-readable artifacts are absent. D2 feasibility, independent
author benefit, full quality, P7/E4 and adoption remain unverified. Stop at the
planned retrospective. Next preparation is 078-B's public candidate/feasibility/
route/leaf contract and exhaustive fixtures; further hardware tests need a new
concrete run freeze and budget. Wider CPU searches remain paused.
