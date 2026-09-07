# 078-B: Real T4 public split composition

All 88 preregistered installed-wheel tests pass at clean source `9ce790e`:
55 new split-operation cases and all 33 storage/aggregation regressions. The exact
JUnit matrix has no failures, errors or skips. Recomputed judging, all 39 snapshot
files against the source commit, 23 installed production modules, 17 pinned package
versions and three result-artifact hashes agree. Raw evidence is in manifest.json,
verdict.json, junit.xml and pytest.log.

## Verified behavior and significance

- Numeric candidates retain the full prepared-data active-bin universe, both
  missing directions, missing-only splits, inactive padding and exact tie order.
- Every active candidate's named child sums, integer counts, scalar gain and
  ordinary/independent feasibility match the frozen original-row float64 oracle.
  Float32 uses unchanged rtol=1e-4/atol=1e-5; counts and routed positions are exact.
- D2 composes ordinary legality with both independent cohort minima on device.
  At minimum=1, the six-row fixture changes from unconstrained threshold 0 with
  gain 12 to constrained threshold 1 with gain 20/3. Renamed/reordered information
  and minima 0/1/2 use the same public operations without task-name dispatch.
- Stable routing preserves unsorted original row positions. Child histograms are
  recomputed from those actual rows; scalar root/child Newton leaves match the
  oracle. Empty, zero-gradient/curvature and no-feasible cases are explicit.
- Caller-bound score/mask buffers retain exact batch identity and borrowed
  lifetime. Inactive padding is excluded even under a supplied all-true mask.
  Unsupported schemas/parameters, nonfinite gains, invalid denominators, negative
  original information/curvature and foreign/forged/released records fail.
- Partial candidate allocation failure preserves inputs and permits later valid
  work. Operations restore the caller stream and retain the private allocation
  scope; previous storage/aggregation checks pass on this same implementation.

Candidate/scoring/mask operations perform no bulk host round trip. Choice exports
one int32 index (4 bytes), and partition exports two int32 child sizes (8 bytes).
Finite/nonnegative validation exports compact flags separately. Tests check these
operation transfers before exporting diagnostic arrays for reference comparison.
This demonstrates device composition for the known D2 change, not independent
author productivity or arbitrary custom-kernel registration.

## Resource observations

| Logged context at minimum=1 | Six-row D2 | Weighted/missing subset | Empty selection |
| --- | ---: | ---: | ---: |
| Upload bytes including explicit preparation | 174 | 256 | 150 |
| Export bytes including reference diagnostics | 1084 | 1288 | 780 |
| Validation bytes within exports | 252 | 252 | 92 |
| Winner/child-size bytes within exports | 24 | 24 | 8 |
| Peak live logical bytes | 1866 | 2524 | 1038 |
| Sampled private-pool peak bytes | 19968 | 19968 | 9216 |
| Explicit synchronizations at logged checkpoint | 89 | 89 | 38 |

The log contains 30 exhaustive-fixture metric records. Their maximum private-pool
peak is 19968 bytes. The separate 8192x32 aggregation regression remains bounded
by the same 16-MiB private-pool cap. Pool metrics exclude CUDA context, driver,
module/JIT and outside allocations; they are not whole-device memory measurements.
The 98 low-occupancy warnings are retained. Pytest reports 8.26 seconds and the
worker 9.75 seconds; these are validation durations, not training benchmarks.
Host dispatch timing includes lazy JIT and is not GPU elapsed time. There is no
end-to-end performance, cost or scaling claim.

## Environment, provenance and allowance

Tesla T4, 15360 MiB, driver 580.95.05. Python 3.12.1 on Linux 4.19/gVisor x86_64,
glibc 2.35. Requested capacity is two CPUs and 8192 MiB host memory; 18 visible
logical CPUs is an environment observation, not the allocated CPU capacity.
The manifest records all pinned versions, including CuPy 13.6.0, numba-cuda 0.27.0,
Numba 0.63.1, llvmlite 0.46.0 and NumPy 2.3.5, with installed import paths.
The image tag is CUDA 12.6.3-devel; the loaded runtime is 12090 and driver API
13000. These are separate observations. Exact CLI, test command, image ID, source
hashes, timestamps and clean Git revision are recorded in manifest.json.

The user explicitly approved uploading the frozen 39-file package to Modal and
one additional T4 invocation: 900-second function, 600-second tests, zero retries.
This run consumed that allowance. The earlier automatic approval block happened
before process creation and did not launch a GPU job. No retry occurred. The
original reproducing command, now blocked by the consumed allowance, was:

```bash
UV_CACHE_DIR=/tmp/openboost-research-uv-cache uv run --no-sync python \
  -m benchmarks.v1.cuda_splits_preflight benchmarks/v1/evidence/cuda-splits-078
```

## Reflection and next boundary

078-B passes its declared real-device acceptance. The foundation now exposes
verified resident statistics, candidate scores, composable feasibility, choice,
routes and scalar leaf values. D2 crosses that public boundary on actual CUDA.

078 overall remains open. Resident targets/derivatives/raw updates, device tree
assembly and prediction, accepted/proposal integration, two-round training and
CPU-readable trained artifacts remain unimplemented. Current recipes are CPU-only.
The 065 run-isolation and 068 retained-state contracts still need testing on the
actual device training path; allocation rollback here is not transaction conformance.
No independent author attempt, full application evaluation, P7/E4 or adoption gate
is passed by these primitive tests. All required R/C/A/E scope remains.

Next is 078-C's explicit accepted/proposal ownership design and independent
two-round scalar fixtures, followed by resident training through these public
operations. Keep D2 independent information in the same path, with saved CPU
inference and rejection/retry checks. Reflect before constructing that next slice;
further hardware execution requires another concrete freeze and allowance.
