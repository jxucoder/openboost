# 2026-09-05: P4.1 fixed-slot histogram primitive

## Context

P3 completed the CPU extension contract. P4.1 needs a device-preserving aggregate
rather than the legacy dictionary wrapper that downloads arrays.

## Decision or Result

Expose HistogramBatch/build_histograms with separate G/H and row counts. Reuse
sample-centric scatter semantics in a small CuPy RawKernel; CPU uses a compiled
sample loop. Existing native kernels use a coupled (..., 2) layout and no public
active mask/count contract, so they are left unchanged. This is a correctness
primitive, not an end-to-end performance change or a new GPU Booster path.

## Changes

- Fixed up to 511 slots, uint8 bins including reserved 255, int32 counts, float32
  G/H, bool active mask, default 256 MiB returned-buffer budget. -1 IDs exclude
  rows; inactive nodes ignore them. Zero-weight rows still count; G/H are already
  weighted. Invalid IDs/dtypes/devices/values and float32 overflow fail explicitly.
- CUDA supports the current CuPy stream and leaves every aggregate on device.
  Validation downloads only scalar booleans; transient validation masks and
  caller input buffers are explicitly outside the histogram budget.
- Standalone real-device suite uses the existing pinned image and wheel-only
  bundle. Direct sample sums provide independent expected values; no production
  histogram constructs expected arrays. Small exact and random/empty cases,
  nondefault stream, separate counts, missing bin, budget and negative paths.

## Verification

- Initial CPU test collection failed because build_histograms was not exported.
- 74 distinct focused tests passed (73-case regression plus added zero-curvature/
  overflow case): batch histogram, runner, experimental objective/dispatch/
  persistence. Changed-file/production lint and docs build passed; only existing
  griffe documentation warnings. GPU evidence remains pending below.
- GPU download spies cover cp.asnumpy, Numba copy_to_host and the legacy wrapper;
  they are not a profiler or proof about arbitrary external calls.

## Failed Attempts

None beyond the initial red test at the time of implementation commit.

## Risks and Follow-ups

- GPU correctness is pending a real T4 run at this implementation commit.
- Atomic summation order is nondeterministic. Scalar validations synchronize.
- Split/routing, leaf reduction and LevelWiseBuilder remain P4.2–P4.4. Existing
  Booster training remains CPU-only; no downstream GPU split/quality claim.

## Commits

- `cfd7dc5` — completed P3 evidence.
